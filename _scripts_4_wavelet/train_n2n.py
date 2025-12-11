"""
NS-N2N Training Script with Plan B Strategy

Plan B: Structure/HU preservation FIRST, noise reduction LATER
- Early epochs (0-30): Strong lambda_rc/hu/edge, weak lambda_noise
- Later epochs (30+): Gradually increase lambda_noise
"""

import sys
import random
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from model_3d_unet_trans import UNet3DTransformer
from losses_n2n import NoiseRemovalLoss, ArtifactRemovalLoss, HighQualityNSN2NLoss
from utils import (
    load_config,
    save_checkpoint,
    load_checkpoint,
    save_simple_samples,
    save_origin_noised_samples,
    EarlyStopping,
)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders with separate datasets"""
    from dataset_n2n import NSN2NDataset  # Local import to avoid circular
    
    nc_ct_dir = config["data"]["nc_ct_dir"]
    hu_window = tuple(config["preprocessing"]["hu_window"])
    patch_size = config["preprocessing"]["patch_size"]
    min_body_fraction = config["preprocessing"]["min_body_fraction"]
    
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    # Slice noise map (for NPS-guided noise augmentation)
    slice_noise_csv = Path(config["dataset"]["slice_noise_csv"])
    slice_noise_map = None
    slice_noise_mean = None
    
    if slice_noise_csv.is_file():
        import pandas as pd
        df_noise = pd.read_csv(slice_noise_csv)
        # Expect columns: patient, z, noise_std (HU)
        slice_noise_map = {
            (str(row["patient"]), int(row["z"])): float(row["noise_std"])
            for _, row in df_noise.iterrows()
        }
        slice_noise_mean = df_noise["noise_std"].mean()
        print(f"[INFO] Loaded slice noise map from {slice_noise_csv}")
        print(f"       Global mean noise_std (HU): {slice_noise_mean:.2f}")
    else:
        print(f"[WARN] slice_noise_csv not found: {slice_noise_csv}")
    
    from dataset_n2n import NSN2NDataset

    # Train dataset (with NPS-guided synthetic noise)
    dataset_train = NSN2NDataset(
        nc_ct_dir=nc_ct_dir,
        hu_window=hu_window,
        patch_size=patch_size,
        min_body_fraction=min_body_fraction,
        lpf_sigma=config["dataset"]["lpf_sigma"],
        lpf_median_size=config["dataset"]["lpf_median_size"],
        match_threshold=config["preprocessing"]["match_threshold"],
        noise_aug_ratio=config["dataset"]["noise_aug_ratio"],
        body_hu_range=tuple(config["dataset"]["body_hu_range"]),
        noise_roi_margin_ratio=config["dataset"]["noise_roi_margin_ratio"],
        noise_tissue_range=tuple(config["dataset"]["noise_tissue_range"]),
        noise_default_std=config["dataset"]["noise_default_std"],
        slice_noise_csv=slice_noise_csv if slice_noise_csv.is_file() else None,
        mode="train",
    )

    # Validation dataset (no synthetic noise, origin only)
    dataset_val = NSN2NDataset(
        nc_ct_dir=nc_ct_dir,
        hu_window=hu_window,
        patch_size=patch_size,
        min_body_fraction=min_body_fraction,
        lpf_sigma=config["dataset"]["lpf_sigma"],
        lpf_median_size=config["dataset"]["lpf_median_size"],
        match_threshold=config["preprocessing"]["match_threshold"],
        noise_aug_ratio=config["dataset"]["noise_aug_ratio"],
        body_hu_range=tuple(config["dataset"]["body_hu_range"]),
        noise_roi_margin_ratio=config["dataset"]["noise_roi_margin_ratio"],
        noise_tissue_range=tuple(config["dataset"]["noise_tissue_range"]),
        noise_default_std=config["dataset"]["noise_default_std"],
        slice_noise_csv=None,
        mode="val",
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def train_epoch(
    model, train_loader, criterion, optimizer, scaler, device, use_amp
):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {
        'rc': 0.0, 'hu': 0.0, 'edge': 0.0,
        'hf_edge': 0.0, 'hf_flat': 0.0, 'syn': 0.0, 'ic': 0.0
    }
    
    for batch in train_loader:
        x_i = batch["x_i"].to(device)
        x_i_aug = batch["x_i_aug"].to(device)
        x_ip1 = batch["x_ip1"].to(device)
        x_mid = batch["x_mid"].to(device)
        W = batch["W"].to(device)
        noise_synthetic = batch["noise_synthetic"].to(device)

        batch_dict = {
            "x_i": x_i,
            "x_ip1": x_ip1,
            "x_mid": x_mid,
            "W": W,
            "noise_synthetic": noise_synthetic,
        }

        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with autocast():
                y_pred, noise_pred = model(x_i_aug)
                loss, comp = criterion(y_pred, noise_pred, batch_dict)
        else:
            y_pred, noise_pred = model(x_i_aug)
            loss, comp = criterion(y_pred, noise_pred, batch_dict)
        
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        for k in loss_components.keys():
            if k in comp:
                loss_components[k] += comp[k].item()
    
    n = len(train_loader)
    total_loss /= n
    for k in loss_components.keys():
        loss_components[k] /= n
    
    return total_loss, loss_components


def validate_epoch(model, val_loader, criterion, device, use_amp):
    """Validation for one epoch"""
    model.eval()
    total_loss = 0.0
    loss_components = {
        'rc': 0.0, 'hu': 0.0, 'edge': 0.0,
        'hf_edge': 0.0, 'hf_flat': 0.0, 'syn': 0.0, 'ic': 0.0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            x_i = batch["x_i"].to(device)
            x_i_aug = batch["x_i_aug"].to(device)
            x_ip1 = batch["x_ip1"].to(device)
            x_mid = batch["x_mid"].to(device)
            W = batch["W"].to(device)
            noise_synthetic = batch["noise_synthetic"].to(device)

            batch_dict = {
                "x_i": x_i,
                "x_ip1": x_ip1,
                "x_mid": x_mid,
                "W": W,
                "noise_synthetic": noise_synthetic,
            }

            if use_amp:
                with autocast():
                    y_pred, noise_pred = model(x_i_aug)
                    loss, comp = criterion(y_pred, noise_pred, batch_dict)
            else:
                y_pred, noise_pred = model(x_i_aug)
                loss, comp = criterion(y_pred, noise_pred, batch_dict)
            
            total_loss += loss.item()
            for k in loss_components.keys():
                if k in comp:
                    loss_components[k] += comp[k].item()
    
    n = len(val_loader)
    total_loss /= n
    for k in loss_components.keys():
        loss_components[k] /= n
    
    return total_loss, loss_components


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train NS-N2N CT Denoising Model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: looks for config/config_n2n.yaml or config_n2n.yaml)')
    args = parser.parse_args()
    
    # Load configuration - try multiple paths
    if args.config:
        config_path = Path(args.config)
    else:
        # Try config folder first, then same directory
        script_dir = Path(__file__).parent
        possible_paths = [
            script_dir / "config" / "config_n2n.yaml",
            script_dir / "config_n2n.yaml",
        ]
        for p in possible_paths:
            if p.is_file():
                config_path = p
                break
        else:
            print("ERROR: No config file specified and no default config found.")
            sys.exit(1)
    
    config = load_config(config_path)
    print(f"[INFO] Loaded config from {config_path}")
    
    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(config.get("seed", 42))
    
    # Create output directory
    output_dir = Path(config["data"]["output_dir"]) / config["experiment"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    model = UNet3DTransformer(
        in_chans=config["model"]["in_chans"],
        img_size=config["model"]["img_size"],
        window_size=config["model"]["window_size"],
        img_range=config["model"]["img_range"],
        depths=config["model"]["depths"],
        embed_dim=config["model"]["embed_dim"],
        num_heads=config["model"]["num_heads"],
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # AMP scaler
    use_amp = config["training"].get("use_amp", True)
    scaler = GradScaler(enabled=use_amp)
    
    # ============================================================
    # LOSS FUNCTION SELECTION
    # - "noise_removal": Stage 1 (random noise removal)
    # - "artifact_removal": Stage 2 (directional streaks, shading)
    # - default: HighQualityNSN2NLoss (legacy)
    # ============================================================
    loss_type = config["loss"].get("loss_type", "default")
    
    if loss_type == "noise_removal":
        print("Using NoiseRemovalLoss (Stage 1: Random Noise Removal)")
        criterion = NoiseRemovalLoss(
            lambda_rc=config["loss"]["lambda_rc"],
            lambda_hu=config["loss"]["lambda_hu"],
            lambda_edge=config["loss"]["lambda_edge"],
            lambda_hf_flat=config["loss"]["lambda_hf_flat"],
            lambda_hf_edge=config["loss"]["lambda_hf_edge"],
            lambda_syn=config["loss"]["lambda_syn"],
            lambda_ic=config["loss"]["lambda_ic"],
            min_body_pixels=config["loss"]["min_body_pixels"],
            artifact_grad_factor=config["loss"]["artifact_grad_factor"],
            flat_threshold=config["loss"]["flat_threshold"],
            hf_target_ratio=config["loss"].get("hf_target_ratio", 0.5),
        ).to(device)
    elif loss_type == "artifact_removal":
        print("Using ArtifactRemovalLoss (Stage 2: Artifact Removal)")
        criterion = ArtifactRemovalLoss(
            lambda_rc=config["loss"]["lambda_rc"],
            lambda_hu=config["loss"]["lambda_hu"],
            lambda_edge=config["loss"]["lambda_edge"],
            lambda_texture=config["loss"]["lambda_texture"],
            lambda_h_streak=config["loss"]["lambda_h_streak"],
            lambda_v_streak=config["loss"]["lambda_v_streak"],
            lambda_lf_artifact=config["loss"]["lambda_lf_artifact"],
            lambda_ic=config["loss"]["lambda_ic"],
            min_body_pixels=config["loss"]["min_body_pixels"],
            artifact_grad_factor=config["loss"]["artifact_grad_factor"],
            flat_threshold=config["loss"]["flat_threshold"],
        ).to(device)
    else:
        print("Using HighQualityNSN2NLoss (Default/Legacy)")
        criterion = HighQualityNSN2NLoss(
            lambda_rc=config["loss"]["lambda_rc"],
            lambda_hu=config["loss"]["lambda_hu"],
            lambda_edge=config["loss"]["lambda_edge"],
            lambda_texture=config["loss"]["lambda_texture"],
            lambda_hf_noise=config["loss"].get("lambda_hf_noise", 1.5),
            lambda_syn=config["loss"].get("lambda_syn", 0.4),
            lambda_ic=config["loss"]["lambda_ic"],
            lambda_mid_noise=config["loss"].get("lambda_mid_noise", 0.8),
            lambda_lf_artifact=config["loss"].get("lambda_lf_artifact", 0.6),
            min_body_pixels=config["loss"]["min_body_pixels"],
            artifact_grad_factor=config["loss"]["artifact_grad_factor"],
            flat_threshold=config["loss"]["flat_threshold"],
        ).to(device)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=config["training"]["min_lr"],
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        delta=config["training"]["early_stopping_delta"],
        verbose=True,
        path=str(ckpt_dir / "best_model.pth"),
    )
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    num_epochs = config["training"]["epochs"]
    global_step = 0
    best_val_loss = float("inf")
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch:03d}/{num_epochs}")
        
        # Train
        train_loss, train_comp = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        # Validate
        val_loss, val_comp = validate_epoch(
            model, val_loader, criterion, device, use_amp
        )
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        
        for k, v in train_comp.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        for k, v in val_comp.items():
            writer.add_scalar(f"Val/{k}", v, epoch)
        
        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "val_loss": val_loss,
            },
            ckpt_dir / f"model_epoch_{epoch:03d}.pth",
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "config": config,
                    "val_loss": val_loss,
                },
                ckpt_dir / "best_model.pth",
            )
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
        
        # Cleanup old checkpoints
        if epoch % (config["training"]["save_interval"] * 3) == 0:
            all_ckpts = sorted(ckpt_dir.glob("model_epoch_*.pth"))
            if len(all_ckpts) > config["training"]["keep_last_n"]:
                for old_ckpt in all_ckpts[:-config["training"]["keep_last_n"]]:
                    old_ckpt.unlink()
    
    writer.close()
    print(f"\n{'='*70}")
    print(f"Training completed! Best val loss: {best_val_loss:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
