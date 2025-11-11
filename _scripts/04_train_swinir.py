"""
SwinIR Fine-tuning for CT Super-Resolution
NC (low-res) -> CE (high-res)
"""
import os
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
from tqdm import tqdm

# SwinIR 경로 추가
SWINIR_PATH = Path(__file__).parent.parent.parent / "SwinIR"
sys.path.insert(0, str(SWINIR_PATH))

try:
    from models.network_swinir import SwinIR
except ImportError:
    print("ERROR: Cannot import SwinIR. Please clone SwinIR repository:")
    print("  git clone https://github.com/JingyunLiang/SwinIR.git")
    sys.exit(1)

# ==================== DATASET ====================
class CTSuperResDataset(Dataset):
    """CT Super-Resolution Dataset"""
    
    def __init__(self, pairs_csv, root_dir, patch_size=64, augment=True):
        self.root = Path(root_dir)
        self.patch_size = patch_size
        self.augment = augment
        
        # Load pairs
        df = pd.read_csv(self.root / pairs_csv)
        self.pairs = []
        
        for _, row in df.iterrows():
            nc_path = row.get('input_nc_norm', '')
            ce_path = row.get('target_ce_norm', '')
            
            if nc_path and ce_path:
                nc_full = self.root / nc_path if not Path(nc_path).is_absolute() else Path(nc_path)
                ce_full = self.root / ce_path if not Path(ce_path).is_absolute() else Path(ce_path)
                
                if nc_full.exists() and ce_full.exists():
                    self.pairs.append((nc_full, ce_full))
        
        print(f"Loaded {len(self.pairs)} valid pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def load_volume(self, path):
        """Load and convert to numpy"""
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # (D, H, W)
        return arr.astype(np.float32)
    
    def extract_2d_patch(self, volume, patch_size):
        """Extract random 2D patch from 3D volume"""
        D, H, W = volume.shape
        
        # Random slice
        d = random.randint(0, D - 1)
        
        # Random patch location
        h_start = random.randint(0, max(0, H - patch_size))
        w_start = random.randint(0, max(0, W - patch_size))
        
        # Extract patch
        patch = volume[d, h_start:h_start+patch_size, w_start:w_start+patch_size]
        
        # Pad if needed
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded = np.zeros((patch_size, patch_size), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
        
        return patch
    
    def augment_patch(self, lr_patch, hr_patch):
        """Data augmentation: flip and rotate"""
        # Random horizontal flip
        if random.random() > 0.5:
            lr_patch = np.fliplr(lr_patch)
            hr_patch = np.fliplr(hr_patch)
        
        # Random vertical flip
        if random.random() > 0.5:
            lr_patch = np.flipud(lr_patch)
            hr_patch = np.flipud(hr_patch)
        
        # Random rotation (90, 180, 270)
        k = random.randint(0, 3)
        lr_patch = np.rot90(lr_patch, k)
        hr_patch = np.rot90(hr_patch, k)
        
        return lr_patch, hr_patch
    
    def __getitem__(self, idx):
        nc_path, ce_path = self.pairs[idx]
        
        # Load volumes
        nc_vol = self.load_volume(nc_path)  # Low-res
        ce_vol = self.load_volume(ce_path)  # High-res
        
        # Extract 2D patches
        lr_patch = self.extract_2d_patch(nc_vol, self.patch_size)
        hr_patch = self.extract_2d_patch(ce_vol, self.patch_size)
        
        # Augmentation
        if self.augment:
            lr_patch, hr_patch = self.augment_patch(lr_patch, hr_patch)
        
        # To tensor (add channel dimension)
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)  # (1, H, W)
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)  # (1, H, W)
        
        return lr_tensor, hr_tensor

# ==================== MODEL ====================
def load_pretrained_swinir(weight_path, upscale=2, in_chans=1):
    """Load pretrained SwinIR and modify for grayscale CT"""
    
    # Model config (same as pretrained)
    model = SwinIR(
        upscale=upscale,
        in_chans=in_chans,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )
    
    # Load pretrained weights
    if Path(weight_path).exists():
        print(f"Loading pretrained weights from: {weight_path}")
        pretrained_dict = torch.load(weight_path, map_location='cpu')
        
        # Handle different weight formats
        if 'params' in pretrained_dict:
            pretrained_dict = pretrained_dict['params']
        elif 'params_ema' in pretrained_dict:
            pretrained_dict = pretrained_dict['params_ema']
        
        # Modify first conv layer if needed (RGB -> Grayscale)
        model_dict = model.state_dict()
        
        if in_chans == 1 and 'conv_first.weight' in pretrained_dict:
            # Average RGB channels to grayscale
            rgb_weight = pretrained_dict['conv_first.weight']  # (C_out, 3, H, W)
            gray_weight = rgb_weight.mean(dim=1, keepdim=True)  # (C_out, 1, H, W)
            pretrained_dict['conv_first.weight'] = gray_weight
        
        # Load weights
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model")
    else:
        print(f"WARNING: Pretrained weight not found at {weight_path}")
        print("Training from scratch...")
    
    return model

# ==================== TRAINING ====================
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.exp_dir = Path(args.exp_dir)
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Datasets
        print("\n" + "="*80)
        print("Loading datasets...")
        self.train_dataset = CTSuperResDataset(
            args.pairs_csv, args.root, 
            patch_size=args.patch_size, augment=True
        )
        
        # Split train/val
        n_train = int(len(self.train_dataset) * 0.9)
        n_val = len(self.train_dataset) - n_train
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [n_train, n_val]
        )
        
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
        
        print(f"Train: {len(self.train_dataset)} samples, Val: {len(self.val_dataset)} samples")
        
        # Model
        print("\nInitializing model...")
        self.model = load_pretrained_swinir(
            args.pretrained_model, 
            upscale=args.upscale, 
            in_chans=1
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=args.lr, 
            betas=(0.9, 0.999)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-7
        )
        
        # Loss
        self.criterion = nn.L1Loss()
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        
        print("="*80 + "\n")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs}")
        for lr, hr in pbar:
            lr, hr = lr.to(self.device), hr.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            sr = self.model(lr)
            
            loss = self.criterion(sr, hr)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for lr, hr in tqdm(self.val_loader, desc="Validating", leave=False):
                lr, hr = lr.to(self.device), hr.to(self.device)
                
                sr = self.model(lr)
                loss = self.criterion(sr, hr)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # Save latest
        latest_path = self.ckpt_dir / 'latest.pth'
        torch.save(ckpt, latest_path)
        
        # Save best
        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(ckpt, best_path)
            print(f"  → Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        # Save periodic
        if epoch % self.args.save_freq == 0:
            epoch_path = self.ckpt_dir / f'epoch_{epoch:04d}.pth'
            torch.save(ckpt, epoch_path)
    
    def train(self):
        print("Starting training...\n")
        
        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{self.args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.2e}")
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LR', lr, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, is_best)
        
        print("\n" + "="*80)
        print(f"Training complete! Best val loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved in: {self.ckpt_dir}")
        print("="*80)
        
        self.writer.close()

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='SwinIR Fine-tuning for CT SR')
    
    # Data
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--exp-dir', default='Outputs/experiments/swinir_x2')
    
    # Model
    parser.add_argument('--pretrained-model', 
                       default=r'Weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth')
    parser.add_argument('--upscale', type=int, default=2, choices=[2, 4])
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-freq', type=int, default=10)
    
    # Device
    parser.add_argument('--gpu', type=int, default=0)
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        print("CUDA not available, using CPU")
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Train
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()