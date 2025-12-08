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
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Import project modules
sys.path.append(str(Path(__file__).parent))
from dataset_n2n import NSN2NDataset
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
    
    # 공통 파라미터
    common_params = {
        "nc_ct_dir": config["data"]["nc_ct_dir"],
        "hu_window": tuple(config["preprocessing"]["hu_window"]),
        "patch_size": config["preprocessing"]["patch_size"],
        "min_body_fraction": config["preprocessing"]["min_body_fraction"],
        "lpf_sigma": config["dataset"]["lpf_sigma"],
        "lpf_median_size": config["dataset"]["lpf_median_size"],
        "match_threshold": config["preprocessing"]["match_threshold"],
        "noise_aug_ratio": config["dataset"]["noise_aug_ratio"],
        "body_hu_range": tuple(config["dataset"]["body_hu_range"]),
        "noise_roi_margin_ratio": config["dataset"]["noise_roi_margin_ratio"],
        "noise_tissue_range": tuple(config["dataset"]["noise_tissue_range"]),
        "noise_default_std": config["dataset"]["noise_default_std"],
    }
    
    # 1. Train Dataset: augmentation ON, synthetic noise ON
    train_dataset = NSN2NDataset(
        **common_params,
        mode="train",  # flip augmentation + synthetic noise
    )
    
    # 2. Validation Dataset: augmentation OFF, clean evaluation
    val_dataset = NSN2NDataset(
        **common_params,
        mode="val",  # no flip, no synthetic noise
    )
    
    # 파일 단위로 train/val 분할
    total_pairs = len(train_dataset.pairs)
    val_size = int(total_pairs * config["training"]["val_split"])
    
    # Seed 기반 셔플
    indices = list(range(total_pairs))
    random.seed(config["training"]["seed"])
    random.shuffle(indices)
    
    # 인덱스 분할
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Pairs 재할당
    original_pairs = train_dataset.pairs.copy()
    train_dataset.pairs = [original_pairs[i] for i in train_indices]
    val_dataset.pairs = [original_pairs[i] for i in val_indices]
    
    print(f"[DATA] Train pairs: {len(train_dataset.pairs)}, Val pairs: {len(val_dataset.pairs)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
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
        'texture': 0.0, 'hf_noise': 0.0, 'syn': 0.0, 'ic': 0.0
    }
    
    for batch_idx, batch in enumerate(train_loader):
        x_i = batch["x_i"].to(device)
        x_i_aug = batch["x_i_aug"].to(device)  # (B, 1, 5, H, W)
        x_ip1 = batch["x_ip1"].to(device)
        x_mid = batch["x_mid"].to(device)
        W = batch["W"].to(device)
        noise_synthetic = batch["noise_synthetic"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with autocast('cuda', enabled=use_amp):
            denoised, noise_pred = model(x_i_aug)
            
            # Prepare batch dict for loss
            batch_dict = {
                "x_i": x_i,
                "x_ip1": x_ip1,
                "x_mid": x_mid,
                "W": W,
                "noise_synthetic": noise_synthetic,
            }
            
            # Calculate loss
            loss, loss_dict = criterion(denoised, noise_pred, batch_dict)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict.get(key, 0.0)
    
    # Average over batches
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches
    
    return avg_loss, loss_components


def validate(model, val_loader, criterion, device, use_amp):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    loss_components = {
        'rc': 0.0, 'hu': 0.0, 'edge': 0.0,
        'texture': 0.0, 'hf_noise': 0.0, 'syn': 0.0, 'ic': 0.0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            x_i = batch["x_i"].to(device)
            x_i_aug = batch["x_i_aug"].to(device)
            x_ip1 = batch["x_ip1"].to(device)
            x_mid = batch["x_mid"].to(device)
            W = batch["W"].to(device)
            noise_synthetic = batch["noise_synthetic"].to(device)
            
            with autocast('cuda', enabled=use_amp):
                denoised, noise_pred = model(x_i_aug)
                
                batch_dict = {
                    "x_i": x_i,
                    "x_ip1": x_ip1,
                    "x_mid": x_mid,
                    "W": W,
                    "noise_synthetic": noise_synthetic,
                }
                
                loss, loss_dict = criterion(denoised, noise_pred, batch_dict)
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict.get(key, 0.0)
    
    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches
    
    return avg_loss, loss_components


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
        config_path = None
        for p in possible_paths:
            if p.exists():
                config_path = p
                break
        
        if config_path is None:
            raise FileNotFoundError(
                f"Config file not found. Tried:\n" + 
                "\n".join(f"  - {p}" for p in possible_paths) +
                "\nPlease specify with --config flag"
            )
    
    config = load_config(str(config_path))
    print(f"Loaded config from: {config_path}")
    
    # Set seed
    set_seed(config["training"]["seed"])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"TRUE Noise Removal: High-Frequency Targeting")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Create output directories
    exp_dir   = Path(config["data"]["output_dir"]) / config["training"]["exp_name"]
    ckpt_dir  = exp_dir / "checkpoints"
    log_dir   = exp_dir / "logs"
    sample_dir = exp_dir / "samples"

    origin_dir = sample_dir / "origin"
    noised_dir = sample_dir / "noised"
    denoise_dir = sample_dir / "denoise"

    for d in [ckpt_dir, log_dir, sample_dir, origin_dir, noised_dir, denoise_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = UNet3DTransformer(
        in_channels=1,
        base_channels=config["model"]["base_channels"],
        num_heads=config["model"]["num_heads"],
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # ============================================================
    # Create loss function based on config
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
            lambda_texture=config["loss"]["lambda_texture"],
            lambda_hf_noise=config["loss"]["lambda_hf_noise"],
            lambda_mid_noise=config["loss"]["lambda_mid_noise"],
            lambda_syn=config["loss"]["lambda_syn"],
            lambda_ic=config["loss"]["lambda_ic"],
            min_body_pixels=config["loss"]["min_body_pixels"],
            artifact_grad_factor=config["loss"]["artifact_grad_factor"],
            flat_threshold=config["loss"]["flat_threshold"],
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


    
    # Save base lambdas (for potential future adjustments)
    base_lambda_rc = criterion.lambda_rc
    base_lambda_hu = criterion.lambda_hu
    base_lambda_edge = criterion.lambda_edge
    base_lambda_texture = criterion.lambda_texture
    base_lambda_hf_noise = criterion.lambda_hf_noise
    base_lambda_ic = criterion.lambda_ic
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        betas=config["training"]["betas"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config["training"]["lr_step_size"],
        gamma=config["training"]["lr_gamma"],
    )
    
    # AMP scaler
    scaler = GradScaler('cuda', enabled=config["training"]["use_amp"])
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        min_delta=config["training"]["early_stopping_delta"],
    )
    
    # Resume from checkpoint if specified
    start_epoch = 1
    if config["training"]["resume"]:
        resume_path = Path(config["training"]["resume"])
        if resume_path.exists():
            start_epoch, _ = load_checkpoint(
                resume_path, model, optimizer, scheduler
            )
            start_epoch += 1
            print(f"Resumed from epoch {start_epoch-1}")
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting training from epoch {start_epoch}")
    print(f"Strategy: TRUE noise removal (high-freq targeting)")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    num_epochs = config["training"]["num_epochs"]
    
    for epoch in range(start_epoch, num_epochs + 1):
        # No dynamic lambda scheduling - new loss targets true noise directly
        
        # Train
        train_loss, train_comps = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, config["training"]["use_amp"]
        )
        
        # Validate
        val_loss, val_comps = validate(
            model, val_loader, criterion, device, config["training"]["use_amp"]
        )
        
        # Step scheduler
        scheduler.step()
        
        # Print progress
        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Lambdas: RC={criterion.lambda_rc:.2f}, HU={criterion.lambda_hu:.2f}, "
              f"Edge={criterion.lambda_edge:.2f}, Texture={criterion.lambda_texture:.2f}, "
              f"HF_Noise={criterion.lambda_hf_noise:.2f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Lambda/hu', criterion.lambda_hu, epoch)
        writer.add_scalar('Lambda/texture', criterion.lambda_texture, epoch)
        writer.add_scalar('Lambda/hf_noise', criterion.lambda_hf_noise, epoch)
        
        for key, val in train_comps.items():
            writer.add_scalar(f'Train/{key}', val, epoch)
        for key, val in val_comps.items():
            writer.add_scalar(f'Val/{key}', val, epoch)
        
        # Save checkpoint
        if epoch % config["training"]["save_interval"] == 0:
            ckpt_path = ckpt_dir / f"model_epoch_{epoch:03d}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, ckpt_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = ckpt_dir / "best_model.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path)
            print(f"  ★ New best model saved! Val Loss: {val_loss:.4f}")
        
        # Save sample images
        if epoch == 1 or epoch % config["training"]["sample_interval"] == 0:
            model.eval()
            with torch.no_grad():
                dataset_full_img = NSN2NDataset(
                    nc_ct_dir=config["data"]["nc_ct_dir"],
                    hu_window=tuple(config["preprocessing"]["hu_window"]),
                    patch_size=0,  # 풀슬라이스
                    min_body_fraction=config["preprocessing"]["min_body_fraction"],
                    lpf_sigma=config["dataset"]["lpf_sigma"],
                    lpf_median_size=config["dataset"]["lpf_median_size"],
                    match_threshold=config["preprocessing"]["match_threshold"],
                    noise_aug_ratio=config["dataset"]["noise_aug_ratio"],  # ★ 훈련이랑 동일 gain
                    body_hu_range=tuple(config["dataset"]["body_hu_range"]),
                    noise_roi_margin_ratio=config["dataset"]["noise_roi_margin_ratio"],
                    noise_tissue_range=tuple(config["dataset"]["noise_tissue_range"]),
                    noise_default_std=config["dataset"]["noise_default_std"],
                    mode="train",   # ★ train 모드 → synthetic noise 들어감
                )
                
                sample_indices = [0, len(dataset_full_img) // 2]

                origin_list = []
                noisy_list = []
                denoised_list = []

                for idx in sample_indices[:2]:
                    sample = dataset_full_img[idx]

                    # Clean center slice (origin)
                    x_clean = sample["x_i"].unsqueeze(0).to(device)          # (1,1,H,W)

                    # 5-slice volume with synthetic noise (input)
                    x_5_aug = sample["x_i_aug"].unsqueeze(0).to(device)      # (1,1,5,H,W)

                    # Extract noisy center slice for visualization
                    # x_i_aug shape: (1, 5, H, W) -> center at index 2
                    x_noisy_center = sample["x_i_aug"][:, 2:3, :, :].to(device)  # (1,1,H,W)

                    with autocast('cuda', enabled=config["training"]["use_amp"]):
                        denoised_out, _ = model(x_5_aug)
                        denoised_out = denoised_out.squeeze(2)  # (1,1,H,W)

                    origin_list.append(x_clean.cpu())
                    noisy_list.append(x_noisy_center.cpu())
                    denoised_list.append(denoised_out.cpu())

                origin_batch = torch.cat(origin_list, dim=0)     # (2,1,H,W)
                noisy_batch = torch.cat(noisy_list, dim=0)       # (2,1,H,W)
                denoised_batch = torch.cat(denoised_list, dim=0) # (2,1,H,W)

                
                print(f"\n  Saving samples for epoch {epoch}:")

                if epoch == 1:
                    print("  → Saving origin & noised reference images (once only):")
                    save_origin_noised_samples(
                        origin=origin_batch,
                        noised=noisy_batch,
                        origin_dir=origin_dir,
                        noised_dir=noised_dir,
                        hu_window=tuple(config["preprocessing"]["hu_window"]),
                        body_hu_range=tuple(config["noise_analysis"]["body_hu_range_roi"]),
                    )
                    print(f"     Saved to: {origin_dir} and {noised_dir}")

                print(f"  → Saving denoised samples for epoch {epoch}:")
                save_simple_samples(
                    noisy=noisy_batch,
                    denoised=denoised_batch,
                    origin_dir=origin_dir,
                    denoise_dir=denoise_dir,
                    epoch=epoch,
                    hu_window=tuple(config["preprocessing"]["hu_window"]),
                    body_hu_range=tuple(config["noise_analysis"]["body_hu_range_roi"]),
                )

        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
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