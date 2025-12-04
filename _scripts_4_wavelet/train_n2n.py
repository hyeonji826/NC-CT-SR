"""
NS-N2N Training Script with Plan B Strategy

Plan B: Structure/HU preservation FIRST, noise reduction LATER
- Early epochs (0-30): Strong lambda_rc/hu/edge, weak lambda_noise
- Later epochs (30+): Gradually increase lambda_noise
"""

import sys
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# Import project modules
sys.path.append(str(Path(__file__).parent))
from dataset_n2n import NSN2NDataset
from model_3d_unet_trans import UNet3DTransformer
from losses_n2n import HighQualityNSN2NLoss
from utils import (
    load_config,
    save_checkpoint,
    load_checkpoint,
    save_simple_samples,
    EarlyStopping,
)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(config: dict):
    """Create train and validation dataloaders"""
    dataset_full = NSN2NDataset(
        nc_ct_dir=config["data"]["nc_ct_dir"],
        hu_window=tuple(config["preprocessing"]["hu_window"]),
        patch_size=config["preprocessing"]["patch_size"],
        min_body_fraction=config["preprocessing"]["min_body_fraction"],
        lpf_sigma=config["dataset"]["lpf_sigma"],
        lpf_median_size=config["dataset"]["lpf_median_size"],
        match_threshold=config["preprocessing"]["match_threshold"],
        noise_aug_ratio=config["dataset"]["noise_aug_ratio"],
        body_hu_range=tuple(config["dataset"]["body_hu_range"]),
        noise_roi_margin_ratio=config["dataset"]["noise_roi_margin_ratio"],
        noise_tissue_range=tuple(config["dataset"]["noise_tissue_range"]),
        noise_default_std=config["dataset"]["noise_default_std"],
        mode="train",
    )
    
    # Split dataset
    total_size = len(dataset_full)
    val_size = int(total_size * config["training"]["val_split"])
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["training"]["seed"])
    )
    
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
        'rc': 0.0, 'noise': 0.0, 'edge': 0.0,
        'hf': 0.0, 'hu': 0.0, 'ic': 0.0
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
        with autocast(enabled=use_amp):
            # Model already outputs denoised image (residual learning inside model)
            denoised = model(x_i_aug)  # (B, 1, 1, H, W)
            
            # Prepare batch dict for loss
            batch_dict = {
                "x_i": x_i,
                "x_ip1": x_ip1,
                "x_mid": x_mid,
                "W": W,
                "noise_synthetic": noise_synthetic,
            }
            
            # Calculate loss
            loss, loss_dict = criterion(denoised, batch_dict)
        
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
        'rc': 0.0, 'noise': 0.0, 'edge': 0.0,
        'hf': 0.0, 'hu': 0.0, 'ic': 0.0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            x_i = batch["x_i"].to(device)
            x_i_aug = batch["x_i_aug"].to(device)
            x_ip1 = batch["x_ip1"].to(device)
            x_mid = batch["x_mid"].to(device)
            W = batch["W"].to(device)
            noise_synthetic = batch["noise_synthetic"].to(device)
            
            with autocast(enabled=use_amp):
                denoised = model(x_i_aug)
                
                batch_dict = {
                    "x_i": x_i,
                    "x_ip1": x_ip1,
                    "x_mid": x_mid,
                    "W": W,
                    "noise_synthetic": noise_synthetic,
                }
                
                loss, loss_dict = criterion(denoised, batch_dict)
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict.get(key, 0.0)
    
    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    for key in loss_components:
        loss_components[key] /= n_batches
    
    return avg_loss, loss_components


def main():
    # Load configuration
    config_path = Path(__file__).parent / "config_n2n.yaml"
    config = load_config(str(config_path))
    
    # Set seed
    set_seed(config["training"]["seed"])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Plan B Training: Structure-First, Then Noise Reduction")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Create output directories
    exp_dir = Path(config["data"]["output_dir"]) / config["training"]["exp_name"]
    ckpt_dir = exp_dir / "checkpoints"
    log_dir = exp_dir / "logs"
    sample_dir = exp_dir / "samples"
    origin_dir = sample_dir / "origin"
    denoise_dir = sample_dir / "denoise"
    
    for d in [ckpt_dir, log_dir, origin_dir, denoise_dir]:
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
    
    # Create loss function
    criterion = HighQualityNSN2NLoss(
        lambda_rc=config["loss"]["lambda_rc"],
        lambda_noise=config["loss"]["lambda_noise"],
        lambda_edge=config["loss"]["lambda_edge"],
        lambda_hf=config["loss"]["lambda_hf"],
        lambda_hu=config["loss"]["lambda_hu"],
        lambda_ic=config["loss"]["lambda_ic"],
        target_noise_ratio=config["loss"]["target_noise_ratio"],
        min_body_pixels=config["loss"]["min_body_pixels"],
        artifact_grad_factor=config["loss"]["artifact_grad_factor"],
        flat_threshold=config["loss"]["flat_threshold"],
    ).to(device)
    
    # Save base lambdas for Plan B scheduling
    base_lambda_rc = criterion.lambda_rc
    base_lambda_hu = criterion.lambda_hu
    base_lambda_edge = criterion.lambda_edge
    base_lambda_noise = criterion.lambda_noise
    base_lambda_hf = criterion.lambda_hf
    base_lambda_ic = criterion.lambda_ic
    noise_warmup_epochs = config["training"]["noise_warmup_epochs"]
    
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
    scaler = GradScaler(enabled=config["training"]["use_amp"])
    
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
    print(f"Plan B Schedule: Warmup {noise_warmup_epochs} epochs (structure-first)")
    print(f"{'='*70}\n")
    
    best_val_loss = float('inf')
    num_epochs = config["training"]["num_epochs"]
    
    for epoch in range(start_epoch, num_epochs + 1):
        # ===== PLAN B: Lambda scheduling =====
        if epoch <= noise_warmup_epochs:
            # Early phase: Emphasize structure/HU, suppress noise term
            criterion.lambda_rc = base_lambda_rc * 1.5
            criterion.lambda_hu = base_lambda_hu * 1.5
            criterion.lambda_edge = base_lambda_edge * 1.2
            criterion.lambda_ic = base_lambda_ic * 1.2
            criterion.lambda_noise = base_lambda_noise * 0.1
            criterion.lambda_hf = base_lambda_hf
            phase = "Structure-First"
        else:
            # Later phase: Gradually increase noise term
            t = min(1.0, (epoch - noise_warmup_epochs) / max(1, num_epochs - noise_warmup_epochs))
            criterion.lambda_noise = base_lambda_noise * (0.3 + 0.7 * t)
            criterion.lambda_rc = base_lambda_rc
            criterion.lambda_hu = base_lambda_hu
            criterion.lambda_edge = base_lambda_edge
            criterion.lambda_ic = base_lambda_ic
            criterion.lambda_hf = base_lambda_hf
            phase = f"Noise-Ramp ({t*100:.0f}%)"
        
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
        print(f"Epoch {epoch:03d}/{num_epochs} [{phase}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Lambdas: RC={criterion.lambda_rc:.2f}, Noise={criterion.lambda_noise:.3f}, "
              f"Edge={criterion.lambda_edge:.2f}, HU={criterion.lambda_hu:.2f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Lambda/noise', criterion.lambda_noise, epoch)
        writer.add_scalar('Lambda/rc', criterion.lambda_rc, epoch)
        
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
        if epoch % config["training"]["sample_interval"] == 0:
            model.eval()
            with torch.no_grad():
                # Get first batch from validation
                sample_batch = next(iter(val_loader))
                x_i = sample_batch["x_i"].to(device)
                x_i_aug = sample_batch["x_i_aug"].to(device)
                
                with autocast(enabled=config["training"]["use_amp"]):
                    denoised = model(x_i_aug).squeeze(2)  # (B, 1, H, W)
                
                # Save samples
                print(f"\n  Saving samples for epoch {epoch}:")
                save_simple_samples(
                    noisy=x_i.cpu(),
                    denoised=denoised.cpu(),
                    origin_dir=origin_dir,
                    denoise_dir=denoise_dir,
                    epoch=epoch,
                    hu_window=tuple(config["preprocessing"]["hu_window"]),
                    body_hu_range=tuple(config["noise_analysis"]["body_hu_range_roi"]),
                )
                print()
        
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