# E:\LD-CT SR\_scripts_4_wavelet\utils.py

import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import warnings
import numpy as np
from skimage.metrics import structural_similarity as ssim


# Suppress warnings
warnings.filterwarnings('ignore')

def calculate_psnr(img1, img2, data_range=1.0):
    """Calculate PSNR between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 10 * np.log10(data_range ** 2 / mse)
    return psnr

def calculate_ssim(img1, img2, data_range=1.0):
    """Calculate SSIM between two images"""
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1.ndim == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            s = ssim(img1[i, 0], img2[i, 0], data_range=data_range)
            ssim_values.append(s)
        return np.mean(ssim_values)
    else:
        return ssim(img1[0], img2[0], data_range=data_range)


def load_config(config_path):
    """Load YAML config with absolute path support"""
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        config_path = script_dir / config_path.name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"Loaded config from: {config_path}")
    return config

def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")
    
    # Save best model
    if is_best:
        best_path = save_path.parent / 'best_model.pth'
        shutil.copyfile(save_path, best_path)
        print(f"Best model saved: {best_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Loaded checkpoint from epoch {epoch}, loss: {loss:.4f}")
    return epoch, loss

def save_sample_images(noisy_input, denoised_output, save_path, epoch):
    """Save sample images for self-supervised learning"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Noisy input
    axes[0].imshow(noisy_input[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Noisy Input (NC-CT)')
    axes[0].axis('off')
    
    # Denoised output
    axes[1].imshow(denoised_output[0, 0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Denoised Output')
    axes[1].axis('off')
    
    plt.suptitle(f'Epoch {epoch} - Self-Supervised Denoising', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def cleanup_old_checkpoints(ckpt_dir, keep_last_n=5):
    """Keep only the last N checkpoints"""
    checkpoints = sorted(ckpt_dir.glob('model_epoch_*.pth'))
    
    if len(checkpoints) > keep_last_n:
        for old_ckpt in checkpoints[:-keep_last_n]:
            if 'best' not in old_ckpt.name:
                old_ckpt.unlink()

class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience=30, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

class WarmupScheduler:
    """Learning rate warmup"""
    def __init__(self, optimizer, warmup_epochs, warmup_lr, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
    
    def is_warmup(self):
        return self.current_epoch < self.warmup_epochs