# train_n2n.py - Self-Supervised Training with Neighbor2Neighbor + Wavelet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import sys
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import yaml

# Add SwinIR to path
sys.path.insert(0, r'E:\LD-CT SR\_externals\SwinIR')
from models.network_swinir import SwinIR

from dataset_n2n import NCCTDenoiseDataset
from losses_n2n import CombinedN2NWaveletLoss, Neighbor2NeighborLoss
from utils import (
    load_config, save_checkpoint, load_checkpoint, save_sample_images,
    cleanup_old_checkpoints, EarlyStopping, WarmupScheduler
)


def train_n2n():
    """
    Self-Supervised Training with Neighbor2Neighbor + Wavelet Sparsity
    
    Key Points:
    1. NO paired data needed!
    2. N2N creates supervision via checkerboard subsampling
    3. Wavelet acts as light regularization
    4. Critical: maintain N2N : Wavelet balance (~20:1)
    """
    
    print("="*80)
    print("🚀 Self-Supervised Training: Neighbor2Neighbor + Wavelet Sparsity")
    print("="*80)
    print("\n📋 Key Features:")
    print("   ✓ NO paired data required (self-supervised!)")
    print("   ✓ N2N: Main denoising mechanism")
    print("   ✓ Wavelet: Regularization (prevents overfitting)")
    print("   ✓ Balance: N2N >> Wavelet (20:1)")
    print("="*80)
    
    # Load config
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config_n2n.yaml'
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    config = load_config(config_path)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Output dirs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config['data']['output_dir']) / f'n2n_wavelet_{timestamp}'
    ckpt_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    sample_dir = exp_dir / 'samples'
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    writer = SummaryWriter(log_dir)
    print(f"\n📊 TensorBoard: tensorboard --logdir={log_dir}")
    
    # Dataset (NC-CT only!)
    print("\n📂 Loading NC-CT dataset (self-supervised)...")
    full_dataset = NCCTDenoiseDataset(
        nc_ct_dir=config['data']['nc_ct_dir'],
        hu_window=config['preprocessing']['hu_window'],
        patch_size=config['preprocessing']['patch_size'],
        config_aug=config['training']['augmentation'],
        mode='train'
    )
    
    val_size = int(len(full_dataset) * config['training']['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Model
    print("\n🏗️  Building SwinIR model...")
    model = SwinIR(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        img_size=config['model']['img_size'],
        window_size=config['model']['window_size'],
        img_range=config['model']['img_range'],
        depths=config['model']['depths'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        upsampler=config['model']['upsampler'],
        resi_connection=config['model']['resi_connection']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Pretrained weights
    pretrained_path = config['model'].get('pretrained')
    if pretrained_path and Path(pretrained_path).exists():
        print(f"\n📥 Loading pretrained weights:")
        print(f"   {pretrained_path}")
        try:
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if 'params' in pretrained_dict:
                pretrained_dict = pretrained_dict['params']
            elif 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']
            
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"   ✅ Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
        except Exception as e:
            print(f"   ⚠️  Failed to load pretrained: {e}")
    
    # Loss function
    print("\n⚖️  Setting up loss function...")
    loss_type = config['training'].get('loss_type', 'n2n_wavelet')
    
    if loss_type == 'n2n_wavelet':
        criterion = CombinedN2NWaveletLoss(
            n2n_gamma=config['training']['n2n_gamma'],
            wavelet_weight=config['training']['wavelet_weight'],
            wavelet_threshold=config['training']['wavelet_threshold'],
            wavelet_levels=config['training']['wavelet_levels']
        ).to(device)
    else:  # n2n_only
        criterion = Neighbor2NeighborLoss(
            gamma=config['training']['n2n_gamma']
        ).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['eta_min']
    )
    
    warmup = WarmupScheduler(
        optimizer,
        warmup_epochs=config['training']['warmup_epochs'],
        warmup_lr=config['training']['warmup_lr'],
        base_lr=config['training']['learning_rate']
    )
    
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )
    
    # Mixed Precision
    use_amp = config['training']['use_amp']
    scaler = GradScaler('cuda') if use_amp else None
    
    # Resume
    start_epoch = 1
    best_val_loss = float('inf')
    if config['training']['resume']:
        resume_path = Path(config['training']['resume'])
        if resume_path.exists():
            start_epoch, _ = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch += 1
            print(f"✅ Resumed from epoch {start_epoch-1}")
    
    # Training Loop
    print("\n" + "="*80)
    print("🚀 Starting Self-Supervised Training...")
    print("="*80 + "\n")
    
    global_step = 0
    
    # Balance monitoring
    balance_warnings = 0
    target_balance = config['monitoring'].get('target_balance_ratio', 20.0)
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        model.train()
        train_losses = []
        train_loss_details = {
            'n2n_rec': [], 'n2n_reg': [], 'n2n_total': [],
            'wavelet': [], 'total': [], 'balance_ratio': []
        }
        
        if warmup.is_warmup():
            warmup.step()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['num_epochs']}")
        
        for batch_idx, noisy_batch in enumerate(pbar):
            # N2N: Only need noisy input! (no target)
            noisy = noisy_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    # N2N loss handles model forward internally
                    loss, loss_dict = criterion(model, noisy)
                
                scaler.scale(loss).backward()
                
                if config['training']['gradient_clip'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['gradient_clip']
                    )
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # N2N loss handles model forward internally
                loss, loss_dict = criterion(model, noisy)
                
                loss.backward()
                
                if config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['gradient_clip']
                    )
                
                optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            for key in train_loss_details:
                if key in loss_dict:
                    train_loss_details[key].append(loss_dict[key])
            
            # TensorBoard logging
            global_step += 1
            writer.add_scalar('Train/total_loss', loss.item(), global_step)
            
            if 'n2n_rec' in loss_dict:
                writer.add_scalar('Train/n2n_rec', loss_dict['n2n_rec'], global_step)
                writer.add_scalar('Train/n2n_reg', loss_dict['n2n_reg'], global_step)
                writer.add_scalar('Train/n2n_total', loss_dict['n2n_total'], global_step)
            
            if 'wavelet_weighted' in loss_dict:
                writer.add_scalar('Train/wavelet', loss_dict['wavelet_weighted'], global_step)
            
            if 'balance_ratio' in loss_dict:
                writer.add_scalar('Train/balance_ratio', loss_dict['balance_ratio'], global_step)
            
            # Update progress bar
            pbar_dict = {
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            }
            if 'n2n_total' in loss_dict:
                pbar_dict['n2n'] = f"{loss_dict['n2n_total']:.4f}"
            if 'wavelet_weighted' in loss_dict:
                pbar_dict['wav'] = f"{loss_dict['wavelet_weighted']:.4f}"
            if 'balance_ratio' in loss_dict:
                pbar_dict['bal'] = f"{loss_dict['balance_ratio']:.1f}"
            
            pbar.set_postfix(pbar_dict)
        
        if not warmup.is_warmup():
            scheduler.step()
        
        # Epoch statistics
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_details = {
            k: sum(v)/len(v) for k, v in train_loss_details.items() if len(v) > 0
        }
        
        # Validation
        model.eval()
        val_losses = []
        val_loss_details = {
            'n2n_rec': [], 'n2n_reg': [], 'n2n_total': [],
            'wavelet': [], 'total': [], 'balance_ratio': []
        }
        
        with torch.no_grad():
            for noisy_batch in tqdm(val_loader, desc="Validation", leave=False):
                noisy = noisy_batch.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast():
                        loss, loss_dict = criterion(model, noisy)
                else:
                    loss, loss_dict = criterion(model, noisy)
                
                val_losses.append(loss.item())
                for key in val_loss_details:
                    if key in loss_dict:
                        val_loss_details[key].append(loss_dict[key])
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_details = {
            k: sum(v)/len(v) for k, v in val_loss_details.items() if len(v) > 0
        }
        
        # Log to TensorBoard
        writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"📊 Epoch {epoch} Summary:")
        print(f"{'='*80}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        if 'n2n_total' in avg_train_details:
            print(f"  ├─ N2N:     {avg_train_details['n2n_total']:.4f}")
            print(f"  │   ├─ Rec: {avg_train_details['n2n_rec']:.4f}")
            print(f"  │   └─ Reg: {avg_train_details['n2n_reg']:.4f}")
        if 'wavelet' in avg_train_details:
            print(f"  └─ Wavelet: {avg_train_details['wavelet']:.4f}")
        
        print(f"\nVal Loss:   {avg_val_loss:.4f}")
        if 'n2n_total' in avg_val_details:
            print(f"  ├─ N2N:     {avg_val_details['n2n_total']:.4f}")
        if 'wavelet' in avg_val_details:
            print(f"  └─ Wavelet: {avg_val_details['wavelet']:.4f}")
        
        # Balance check
        if 'balance_ratio' in avg_train_details:
            ratio = avg_train_details['balance_ratio']
            print(f"\n⚖️  Balance Ratio: {ratio:.2f} (target: ~{target_balance:.1f})")
            
            if config['monitoring'].get('warn_if_imbalanced', True):
                if ratio < target_balance * 0.5:
                    print(f"   ⚠️  WARNING: Wavelet too strong! (ratio < {target_balance*0.5:.1f})")
                    balance_warnings += 1
                elif ratio > target_balance * 2.0:
                    print(f"   ⚠️  WARNING: Wavelet too weak! (ratio > {target_balance*2.0:.1f})")
                    balance_warnings += 1
                else:
                    print(f"   ✅ Balance is good!")
        
        print(f"\nLR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}\n")
        
        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        if epoch % config['training']['save_interval'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_val_loss,
                ckpt_dir / f"model_epoch_{epoch}.pth",
                is_best=is_best
            )
            cleanup_old_checkpoints(ckpt_dir, config['training']['keep_last_n'])
        
        # Save samples
        if epoch % config['training']['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                noisy_sample = next(iter(val_loader))
                noisy_sample = noisy_sample.to(device)
                
                # Get denoised output
                if use_amp:
                    with autocast():
                        denoised = model(noisy_sample)
                else:
                    denoised = model(noisy_sample)
                
                denoised = torch.clamp(denoised, 0, 1)
                
                # Save comparison (noisy vs denoised vs difference)
                save_sample_images(
                    noisy_sample, denoised, noisy_sample,  # Use noisy as "target" for visualization
                    sample_dir / f"epoch_{epoch}.png",
                    epoch
                )
        
        # Early stopping
        if early_stopping(avg_val_loss):
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            print(f"   Best val loss: {best_val_loss:.4f}")
            break
    
    writer.close()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ Training Completed!")
    print("="*80)
    print(f"📁 Experiment directory: {exp_dir}")
    print(f"📁 Checkpoints: {ckpt_dir}")
    print(f"📁 Samples: {sample_dir}")
    print(f"📊 Best val loss: {best_val_loss:.4f}")
    print(f"📊 TensorBoard: tensorboard --logdir={log_dir}")
    
    if balance_warnings > 0:
        print(f"\n⚠️  Balance warnings: {balance_warnings}")
        print(f"   Consider adjusting wavelet_weight in config")
    
    print("="*80)


if __name__ == '__main__':
    train_n2n()