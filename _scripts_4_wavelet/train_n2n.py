"""
NS-N2N Training Script
Stage 1: Noise Removal (random high-freq grain)
Stage 2: Artifact Removal (directional streaks, shading)
"""

import sys
import random
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from model_3d_unet_trans import UNet3DTransformer
from losses_n2n import NoiseRemovalLoss, ArtifactRemovalLoss
from dataset_n2n import NSN2NDataset
from utils import (
    load_config,
    save_checkpoint,
    load_checkpoint,
    save_simple_samples,
    save_origin_noised_samples,
    EarlyStopping,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(config: dict):
    """Create train/val dataloaders"""
    nc_ct_dir = config["data"]["nc_ct_dir"]
    hu_window = tuple(config["preprocessing"]["hu_window"])
    patch_size = config["preprocessing"]["patch_size"]
    min_body_fraction = config["preprocessing"]["min_body_fraction"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    slice_noise_csv = config["dataset"].get("slice_noise_csv", None)

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
        slice_noise_csv=slice_noise_csv,
        mode="train",
        augment_streaks=config["dataset"].get("augment_streaks", False),
        streak_strength=config["dataset"].get("streak_strength", 0.1),
    )

    # Validation dataset (no synthetic noise, origin only)
    dataset_val = NSN2NDataset(
        nc_ct_dir=nc_ct_dir,
        hu_window=hu_window,
        patch_size=0,  # No crop for validation
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
        batch_size=2,  # Fixed: HN/LN pair
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {}
    
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
        for k, v in comp.items():
            loss_components[k] = loss_components.get(k, 0.0) + v.item()
    
    n = len(train_loader)
    total_loss /= n
    for k in loss_components.keys():
        loss_components[k] /= n
    
    return total_loss, loss_components


def validate_epoch(
    model, val_loader, criterion, device, use_amp,
    sample_dir, origin_dir, noised_dir, epoch,
    hu_window, body_hu_range, save_samples=False
):
    """Validation for one epoch with sample saving"""
    model.eval()
    total_loss = 0.0
    loss_components = {}
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
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
            for k, v in comp.items():
                loss_components[k] = loss_components.get(k, 0.0) + v.item()
            
            # Save first batch samples
            if save_samples and i == 0:
                # origin: x_i (clean center slice, no flip)
                # noised: x_i_aug center slice (synthetic noise added, no flip)
                # denoised: y_pred (model output)
                x_i_center = x_i_aug[:, :, 2:3, :, :]  # (B, 1, 1, H, W)
                save_simple_samples(
                    origin=x_i.squeeze(2),
                    noisy=x_i_center.squeeze(2),
                    denoised=y_pred.squeeze(2),
                    origin_dir=origin_dir,
                    denoise_dir=sample_dir,
                    epoch=epoch,
                    hu_window=hu_window,
                    body_hu_range=body_hu_range,
                )
    
    n = len(val_loader)
    total_loss /= n
    for k in loss_components.keys():
        loss_components[k] /= n
    
    return total_loss, loss_components


def main():
    parser = argparse.ArgumentParser(description='Train NS-N2N CT Denoising Model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    print(f"[INFO] Loaded config from {args.config}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(config["training"].get("seed", 42))
    
    # Output directory: stage-specific with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config["training"]["exp_name"]
    output_dir = Path(config["data"]["output_dir"]) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    sample_dir = output_dir / "sample" / "denoise"
    sample_dir.mkdir(parents=True, exist_ok=True)
    origin_dir = output_dir / "sample" / "origin"
    origin_dir.mkdir(parents=True, exist_ok=True)
    noised_dir = output_dir / "sample" / "noise"
    noised_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config copy
    import shutil
    shutil.copy(args.config, output_dir / "config.yaml")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Model
    model = UNet3DTransformer(
        in_channels=1,
        base_channels=config["model"]["base_channels"],
        num_heads=config["model"]["num_heads"],
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=config["training"]["betas"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # AMP scaler
    use_amp = config["training"].get("use_amp", True)
    scaler = GradScaler(enabled=use_amp)
    
    # Loss function selection
    loss_type = config["loss"]["loss_type"]
    
    if loss_type == "noise_removal":
        print("[INFO] Using NoiseRemovalLoss (Stage 1)")
        criterion = NoiseRemovalLoss(
            lambda_rc=config["loss"]["lambda_rc"],
            lambda_hu=config["loss"]["lambda_hu"],
            lambda_edge=config["loss"]["lambda_edge"],
            lambda_hf_edge=config["loss"]["lambda_hf_edge"],
            lambda_hf_flat=config["loss"]["lambda_hf_flat"],
            lambda_syn=config["loss"]["lambda_syn"],
            lambda_ic=config["loss"]["lambda_ic"],
            min_body_pixels=config["loss"]["min_body_pixels"],
            artifact_grad_factor=config["loss"]["artifact_grad_factor"],
            flat_threshold=config["loss"]["flat_threshold"],
            hf_target_ratio=config["loss"]["hf_target_ratio"],
            edge_threshold=config["loss"].get("edge_threshold", 0.05),
        ).to(device)
    elif loss_type == "artifact_removal":
        print("[INFO] Using ArtifactRemovalLoss (Stage 2)")
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
            edge_threshold=config["loss"].get("edge_threshold", 0.05),
        ).to(device)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config["training"]["lr_gamma"],
        patience=config["training"]["lr_patience"],
        verbose=True,
    )
    
    # Resume or load pretrained
    start_epoch = 1
    resume_path = config["training"].get("resume", None)
    if resume_path and Path(resume_path).is_file():
        print(f"[INFO] Loading checkpoint from {resume_path}")
        epoch, loss = load_checkpoint(Path(resume_path), model, optimizer, scheduler)
        start_epoch = epoch + 1
        print(f"[INFO] Resumed from epoch {epoch} (loss: {loss:.4f})")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        min_delta=config["training"]["early_stopping_delta"],
    )
    
    # Training loop
    num_epochs = config["training"]["num_epochs"]
    best_val_loss = float("inf")
    
    # Save origin/noised samples once (epoch 0)
    print("\n[INFO] Saving origin/noised samples...")
    model.eval()
    with torch.no_grad():
        # Use train_loader to get synthetic noise
        train_iter = iter(train_loader)
        batch = next(train_iter)
        
        x_i = batch["x_i"].to(device)
        x_i_aug = batch["x_i_aug"].to(device)
        x_i_center = x_i_aug[:, :, 2:3, :, :]
        
        # Take first 2 samples (HN/LN)
        save_origin_noised_samples(
            origin=x_i[:2].squeeze(2),
            noised=x_i_center[:2].squeeze(2),
            origin_dir=origin_dir,
            noised_dir=noised_dir,
            hu_window=tuple(config["preprocessing"]["hu_window"]),
            body_hu_range=tuple(config["dataset"]["body_hu_range"]),
        )
    
    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch:03d}/{num_epochs}")
        
        train_loss, train_comp = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        
        save_samples = (epoch % config["training"]["sample_interval"] == 0)
        val_loss, val_comp = validate_epoch(
            model, val_loader, criterion, device, use_amp,
            sample_dir, origin_dir, noised_dir, epoch,
            tuple(config["preprocessing"]["hu_window"]),
            tuple(config["dataset"]["body_hu_range"]),
            save_samples=save_samples,
        )
        
        scheduler.step(val_loss)
        
        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        
        for k, v in train_comp.items():
            writer.add_scalar(f"Train/{k}", v, epoch)
        for k, v in val_comp.items():
            writer.add_scalar(f"Val/{k}", v, epoch)
        
        # Save checkpoint
        if epoch % config["training"]["save_interval"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                ckpt_dir / f"model_epoch_{epoch:03d}.pth"
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss,
                ckpt_dir / "best_model.pth"
            )
            print(f"  [BEST] Saved best model (val_loss: {val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_loss):
            print("Early stopping triggered!")
            break
        
        # Cleanup old checkpoints
        if epoch % (config["training"]["save_interval"] * 3) == 0:
            all_ckpts = sorted(ckpt_dir.glob("model_epoch_*.pth"))
            keep_last_n = config["training"]["keep_last_n"]
            if len(all_ckpts) > keep_last_n:
                for old_ckpt in all_ckpts[:-keep_last_n]:
                    old_ckpt.unlink()
    
    writer.close()
    print(f"\n{'='*70}")
    print(f"Training completed! Best val loss: {best_val_loss:.4f}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()