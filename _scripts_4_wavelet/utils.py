"""
Utility functions for NS-N2N training
"""

import torch
import yaml
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from typing import Tuple


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    torch.save(checkpoint, save_path)


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
    return epoch, loss


def compute_noise_hu(
    x_01: torch.Tensor,
    hu_window: Tuple[float, float],
    body_hu_range: Tuple[float, float],
    roi_h: float = 0.6,
    roi_w: float = 0.6,
) -> float:
    """
    Compute noise in HU using high-pass residual (Plan B)
    
    This prevents "scale-down cheating" by:
    1. Removing low-frequency structure via Gaussian filter
    2. Measuring std only on high-pass residual
    3. True noise reduction shows in HP residual, not just global darkening
    
    Args:
        x_01: Tensor in [0, 1] range, shape (B, C, H, W)
        hu_window: (min, max) HU window for denormalization
        body_hu_range: (min, max) HU range for body mask
        roi_h, roi_w: ROI size as fraction of image
    
    Returns:
        noise_std_hu: Noise level in HU units
    """
    hu_min, hu_max = hu_window
    arr = x_01[0, 0].detach().cpu().numpy()
    H, W = arr.shape
    
    # Extract central ROI
    h_margin = int(H * (1.0 - roi_h) / 2.0)
    w_margin = int(W * (1.0 - roi_w) / 2.0)
    roi = arr[h_margin:H - h_margin, w_margin:W - w_margin]
    
    # Convert to HU
    roi_hu = roi * (hu_max - hu_min) + hu_min
    
    # Body mask
    body_mask = (roi_hu >= body_hu_range[0]) & (roi_hu <= body_hu_range[1])
    if body_mask.sum() < 100:
        return 0.0
    
    # High-pass filter: Remove structure, keep only noise
    lp = gaussian_filter(roi_hu, sigma=1.0)
    hp = roi_hu - lp
    
    # Measure std on high-pass residual only
    noise_std_hu = float(hp[body_mask].std())
    return noise_std_hu


def save_simple_samples(
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    origin_dir: Path,
    denoise_dir: Path,
    epoch: int,
    hu_window: Tuple[float, float],
    body_hu_range: Tuple[float, float],
):
    """
    Save samples as individual images in separate folders (Plan B)
    
    Structure:
        origin/
            epoch_10_HN.png
            epoch_10_LN.png
        denoise/
            epoch_10_HN.png  (with noise HU in filename would be ideal, but keeping simple)
            epoch_10_LN.png
    
    Args:
        noisy: (B, 1, H, W) noisy input
        denoised: (B, 1, H, W) denoised output
        origin_dir: Directory for noisy images
        denoise_dir: Directory for denoised images
        epoch: Current epoch number
        hu_window: HU window for noise calculation
        body_hu_range: Body HU range for noise calculation
    """
    origin_dir.mkdir(parents=True, exist_ok=True)
    denoise_dir.mkdir(parents=True, exist_ok=True)
    
    # Process first 2 samples (HN and LN)
    labels = ['HN', 'LN']
    for idx in range(min(2, noisy.shape[0])):
        label = labels[idx]
        
        # Extract and rotate images
        noisy_np = noisy[idx, 0].cpu().numpy()
        denoised_np = denoised[idx, 0].detach().cpu().numpy()
        
        noisy_np = np.rot90(noisy_np, k=1)
        denoised_np = np.rot90(denoised_np, k=1)
        
        # Calculate noise HU for console logging
        noisy_hu = compute_noise_hu(
            noisy[idx:idx+1], hu_window, body_hu_range
        )
        denoised_hu = compute_noise_hu(
            denoised[idx:idx+1], hu_window, body_hu_range
        )
        
        # Print to console
        reduction = ((noisy_hu - denoised_hu) / noisy_hu * 100) if noisy_hu > 0 else 0
        print(f"  [{label}] Original: {noisy_hu:.1f} HU → Denoised: {denoised_hu:.1f} HU ({reduction:.1f}% reduction)")
        
        # Save origin
        plt.figure(figsize=(6, 6))
        plt.imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout(pad=0)
        origin_path = origin_dir / f"epoch_{epoch}_{label}.png"
        plt.savefig(origin_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Save denoised
        plt.figure(figsize=(6, 6))
        plt.imshow(denoised_np, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout(pad=0)
        denoise_path = denoise_dir / f"epoch_{epoch}_{label}.png"
        plt.savefig(denoise_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()


class EarlyStopping:
    """Early stopping handler"""
    def __init__(self, patience: int = 50, min_delta: float = 0.00003):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
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