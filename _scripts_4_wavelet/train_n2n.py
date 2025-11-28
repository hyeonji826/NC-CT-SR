import warnings
warnings.filterwarnings("ignore")

import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import sys

sys.path.insert(0, r"E:\LD-CT SR\_externals\SwinIR")
from models.network_swinir import SwinIR

from dataset_n2n import NCCTDenoiseDataset
from losses_n2n import WeightedLoss
from utils import (
    load_yaml_config, save_checkpoint, load_checkpoint,
    save_sample_images, cleanup_old_checkpoints, EarlyStopping
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    loss_dict_sum = {}
    
    pbar = tqdm(loader, desc="Training")
    for inputs, targets, weights in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss, loss_dict = criterion(outputs, targets, weights)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_dict_sum[k] = loss_dict_sum.get(k, 0.0) + v
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(loader)
    avg_loss_dict = {k: v / len(loader) for k, v in loss_dict_sum.items()}
    return avg_loss, avg_loss_dict


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    for inputs, targets, weights in tqdm(loader, desc="Validation"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        weights = weights.to(device)
        
        outputs = model(inputs)
        loss, _ = criterion(outputs, targets, weights)
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    set_seed(42)
    
    # Load config
    config_path = Path(r"E:\LD-CT SR\_scripts_4_wavelet\config\config_n2n.yaml")
    cfg = load_yaml_config(str(config_path))
    
    output_dir = Path(cfg['data']['output_dir']) / 'ns_n2n_optimized'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / 'ckpts'
    ckpt_dir.mkdir(exist_ok=True)
    sample_dir = output_dir / 'samples'
    sample_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Device: {device}")
    
    # Dataset
    full_dataset = NCCTDenoiseDataset(
        nc_ct_dir=cfg['data']['nc_ct_dir'],
        hu_window=tuple(cfg['preprocessing']['hu_window']),
        patch_size=cfg['preprocessing']['patch_size'],
        mode='train'
    )
    
    val_size = int(len(full_dataset) * cfg['training']['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )
    
    print(f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model - SwinIR 1채널 출력
    model = SwinIR(
        upscale=1,
        in_chans=3,
        img_size=cfg['preprocessing']['patch_size'],
        window_size=cfg['model']['window_size'],
        img_range=1.0,
        depths=cfg['model']['depths'],
        embed_dim=cfg['model']['embed_dim'],
        num_heads=cfg['model']['num_heads'],
        mlp_ratio=cfg['model']['mlp_ratio'],
        upsampler=None,
        resi_connection='1conv'
    ).to(device)
    
    # 출력 1채널로 수정
    if hasattr(model, 'conv_last'):
        model.conv_last = nn.Conv2d(
            cfg['model']['embed_dim'], 1, 3, 1, 1
        ).to(device)
    
    # Pretrained weights
    pretrained_path = Path(cfg['model'].get('pretrained_path', ''))
    if pretrained_path.exists():
        try:
            pretrained = torch.load(str(pretrained_path), map_location='cpu')
            if 'params' in pretrained:
                pretrained = pretrained['params']
            
            # conv_last 제외하고 로드
            model_dict = model.state_dict()
            pretrained = {k: v for k, v in pretrained.items() 
                         if k in model_dict and 'conv_last' not in k 
                         and model_dict[k].shape == v.shape}
            model_dict.update(pretrained)
            model.load_state_dict(model_dict)
            print(f"✅ Loaded pretrained weights (partial)")
        except Exception as e:
            print(f"⚠️ Pretrained load failed: {e}")
    
    # Loss
    criterion = WeightedLoss().to(device)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        betas=tuple(cfg['training']['betas']),
        weight_decay=cfg['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    
    scaler = GradScaler()
    early_stopping = EarlyStopping(
        patience=cfg['training']['early_stopping_patience'],
        min_delta=cfg['training']['early_stopping_delta']
    )
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    resume_path_str = cfg['training'].get('resume', '')
    if resume_path_str:
        resume_path = Path(resume_path_str)
        if resume_path.exists():
            start_epoch, _ = load_checkpoint(resume_path, model, optimizer, scheduler)
            print(f"✅ Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n🏋️ Training start...\n")
    for epoch in range(start_epoch, cfg['training']['num_epochs']):
        print(f"Epoch {epoch + 1}/{cfg['training']['num_epochs']}")
        
        train_loss, train_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print(f"  MSE: {train_loss_dict['mse']:.4f}, Matched: {train_loss_dict['matched_ratio']*100:.1f}%")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        if (epoch + 1) % cfg['training']['save_interval'] == 0 or is_best:
            save_path = ckpt_dir / f"model_epoch_{epoch + 1}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, save_path, is_best)
            cleanup_old_checkpoints(ckpt_dir, keep_last_n=cfg['training']['keep_last_n'])
        
        # Save samples
        if (epoch + 1) % cfg['training']['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                sample_inputs, _ = next(iter(val_loader))
                sample_inputs = sample_inputs[:2].to(device)
                sample_outputs = model(sample_inputs)
                
                save_sample_images(
                    sample_inputs[:, 1:2].cpu(),  # 중앙 채널 입력
                    sample_outputs.cpu(),  # 1채널 출력
                    sample_dir / f"epoch_{epoch + 1}.png",
                    epoch + 1
                )
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
            break
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()