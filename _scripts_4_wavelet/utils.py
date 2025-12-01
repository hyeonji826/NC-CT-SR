# E:\LD-CT SR\_scripts_4_wavelet\utils.py

import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
import warnings
import numpy as np
from skimage.metrics import structural_similarity as ssim

import yaml

def load_yaml_config(path: str):
    """
    Load YAML config file safely.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

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

def save_sample_images(noisy_input, denoised_output, save_path, epoch, metrics=None):
    """HN + LN 비교 (2열: Noisy, Denoised)"""
    num_samples = noisy_input.shape[0]
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 6*num_samples + 1))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples):
        noisy_np = noisy_input[idx, 0].cpu().numpy()
        denoised_np = denoised_output[idx, 0].detach().cpu().numpy()
        
        noisy_np = np.rot90(noisy_np, k=1)
        denoised_np = np.rot90(denoised_np, k=1)
        
        if metrics and idx < len(metrics):
            m = metrics[idx]
            label = m.get('label', f'Sample {idx+1}')
            orig_noise = m.get('original_noise_hu', 0)
            final_noise = m.get('estimated_noise_hu', 0)
        else:
            label = f'Sample {idx+1}'
            tissue_mask = (noisy_np > 0.2) & (noisy_np < 0.8)
            if tissue_mask.sum() > 100:
                orig_noise = noisy_np[tissue_mask].std() * 400
                final_noise = denoised_np[tissue_mask].std() * 400
            else:
                orig_noise = final_noise = 0
        
        label_color = 'red' if label == 'HN' else 'blue'
        
        # Noisy
        axes[idx, 0].imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
        title_str = f'[{label}] Noisy\nNoise: {orig_noise:.1f} HU'
        axes[idx, 0].set_title(title_str, fontsize=12, fontweight='bold', color=label_color)
        axes[idx, 0].axis('off')
        
        # Denoised
        axes[idx, 1].imshow(denoised_np, cmap='gray', vmin=0, vmax=1)
        reduction = ((orig_noise - final_noise) / orig_noise * 100) if orig_noise > 0 else 0
        title_str = f'[{label}] Denoised (Epoch {epoch})\nNoise: {final_noise:.1f} HU ({reduction:.1f}% ↓)'
        axes[idx, 1].set_title(title_str, fontsize=12, fontweight='bold', color=label_color)
        axes[idx, 1].axis('off')
    
    title = f'Epoch {epoch} - Enhanced NS-N2N'
    if metrics and len(metrics) >= 2:
        hn_noise = metrics[0].get('estimated_noise_hu', 0)
        ln_noise = metrics[1].get('estimated_noise_hu', 0)
        if hn_noise > 0 and ln_noise > 0:
            title += f'\nFinal - HN: {hn_noise:.1f} HU | LN: {ln_noise:.1f} HU'
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
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