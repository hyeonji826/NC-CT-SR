import torch
import yaml
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from typing import Tuple


# -------------------------------------------------------------------------
# HU conversion and windowing
# -------------------------------------------------------------------------

def denormalize_to_hu(x_norm: np.ndarray, hu_min: float = -160, hu_max: float = 240) -> np.ndarray:
    """Convert normalized [0,1] to HU range"""
    return x_norm * (hu_max - hu_min) + hu_min


def apply_window(hu_img: np.ndarray, wl: float = 40, ww: float = 400) -> np.ndarray:
    """Apply CT windowing for visualization"""
    lo = wl - ww // 2
    hi = wl + ww // 2
    hu_img_clip = np.clip(hu_img, lo, hi)
    return (hu_img_clip - lo) / (hi - lo)


# -------------------------------------------------------------------------
# Config / checkpoint
# -------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scheduler, epoch, loss, save_path: Path):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path: Path, model, optimizer=None, scheduler=None):
    """Load checkpoint and restore model / optimizer / scheduler."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        if checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    return epoch, loss


# -------------------------------------------------------------------------
# Noise metric (HU)
# -------------------------------------------------------------------------

def compute_noise_hu(
    x_01: torch.Tensor,
    hu_window: Tuple[float, float],
    body_hu_range: Tuple[float, float],
    roi_h: float = 0.6,
    roi_w: float = 0.6,
    use_highpass: bool = True,
    debug: bool = False,
) -> float:
    hu_min, hu_max = hu_window
    arr = x_01[0, 0].detach().cpu().numpy()
    H, W = arr.shape

    # Center ROI
    h_margin = int(H * (1.0 - roi_h) / 2.0)
    w_margin = int(W * (1.0 - roi_w) / 2.0)
    roi = arr[h_margin : H - h_margin, w_margin : W - w_margin]

    # Body mask
    body_mask_norm = (roi > 0.15) & (roi < 0.85)
    if body_mask_norm.sum() < 100:
        return 0.0

    # Convert to HU
    roi_hu = roi * (hu_max - hu_min) + hu_min

    if use_highpass:
        # High-pass filtering to measure only noise
        lp = gaussian_filter(roi_hu, sigma=1.0)
        hp = roi_hu - lp
        noise_std_hu = float(hp[body_mask_norm].std())
        
        if debug:
            raw_std_hu = float(roi_hu[body_mask_norm].std())
            ratio = noise_std_hu / raw_std_hu if raw_std_hu > 0 else 0.0
            print(
                f"    [DEBUG] Raw std: {raw_std_hu:.1f} HU | "
                f"HP std: {noise_std_hu:.1f} HU | Ratio: {ratio:.2%}"
            )
    else:
        noise_std_hu = float(roi_hu[body_mask_norm].std())
        if debug:
            print(f"    [DEBUG] Raw std: {noise_std_hu:.1f} HU (no HP filtering)")

    return noise_std_hu


# -------------------------------------------------------------------------
# Sample saving
# -------------------------------------------------------------------------

def save_simple_samples(
    origin: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    origin_dir: Path,
    denoise_dir: Path,
    epoch: int,
    hu_window: Tuple[float, float],
    body_hu_range: Tuple[float, float],
):
    denoise_dir.mkdir(parents=True, exist_ok=True)

    labels = ["HN", "LN"]
    for idx in range(min(2, noisy.shape[0])):
        label = labels[idx]

        # Normalized arrays
        origin_norm = origin[idx, 0].cpu().numpy()
        noisy_norm = noisy[idx, 0].cpu().numpy()
        denoised_norm = denoised[idx, 0].detach().cpu().numpy()

        # Convert to HU
        hu_min, hu_max = hu_window
        origin_hu = denormalize_to_hu(origin_norm, hu_min, hu_max)
        noisy_hu = denormalize_to_hu(noisy_norm, hu_min, hu_max)
        denoised_hu = denormalize_to_hu(denoised_norm, hu_min, hu_max)

        # Apply windowing for visualization
        origin_win = apply_window(origin_hu, wl=40, ww=400)
        noisy_win = apply_window(noisy_hu, wl=40, ww=400)
        denoised_win = apply_window(denoised_hu, wl=40, ww=400)

        # Rotate for display
        origin_win = np.rot90(origin_win, k=1)
        noisy_win = np.rot90(noisy_win, k=1)
        denoised_win = np.rot90(denoised_win, k=1)

        # Compute noise (high-pass)
        origin_hu_std = compute_noise_hu(
            origin[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=True,
            debug=False,
        )
        noisy_hu_std = compute_noise_hu(
            noisy[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=True,
            debug=False,
        )
        denoised_hu_std = compute_noise_hu(
            denoised[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=True,
            debug=False,
        )

        reduction_from_noisy = (
            (noisy_hu_std - denoised_hu_std) / noisy_hu_std * 100 if noisy_hu_std > 0 else 0.0
        )
        reduction_from_origin = (
            (origin_hu_std - denoised_hu_std) / origin_hu_std * 100 if origin_hu_std > 0 else 0.0
        )
        
        print(
            f"  [{label}] Origin: {origin_hu_std:.1f} HU → "
            f"Noisy: {noisy_hu_std:.1f} HU → "
            f"Denoised: {denoised_hu_std:.1f} HU "
            f"(vs Noisy: {reduction_from_noisy:.1f}% ↓, vs Origin: {reduction_from_origin:+.1f}%)"
        )

        # Figure size
        h, w = noisy_win.shape
        aspect_ratio = w / h
        fig_height = 8
        fig_width = fig_height * aspect_ratio * 3  # 3 images side by side

        # Save by label
        label_dir = denoise_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # Save 3-panel comparison
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        
        axes[0].imshow(origin_win, cmap="gray", vmin=0, vmax=1)
        axes[0].axis("off")
        axes[0].set_title(f"Origin - {origin_hu_std:.1f} HU", fontsize=14, pad=10)
        
        axes[1].imshow(noisy_win, cmap="gray", vmin=0, vmax=1)
        axes[1].axis("off")
        axes[1].set_title(f"Noisy - {noisy_hu_std:.1f} HU", fontsize=14, pad=10)
        
        axes[2].imshow(denoised_win, cmap="gray", vmin=0, vmax=1)
        axes[2].axis("off")
        axes[2].set_title(
            f"Denoised - {denoised_hu_std:.1f} HU ({reduction_from_origin:+.1f}%)",
            fontsize=14,
            pad=10,
        )
        
        plt.tight_layout()
        compare_path = label_dir / f"epoch_{epoch:03d}_comparison.png"
        plt.savefig(compare_path, dpi=150, bbox_inches="tight")
        plt.close()


def save_origin_noised_samples(
    origin: torch.Tensor,
    noised: torch.Tensor,
    origin_dir: Path,
    noised_dir: Path,
    hu_window: Tuple[float, float],
    body_hu_range: Tuple[float, float],
):
    """Save origin (clean NC) and synthetic noised images once"""
    origin_dir.mkdir(parents=True, exist_ok=True)
    noised_dir.mkdir(parents=True, exist_ok=True)

    labels = ["HN", "LN"]
    for idx in range(min(2, origin.shape[0])):
        label = labels[idx]

        # Normalized arrays
        origin_norm = origin[idx, 0].cpu().numpy()
        noised_norm = noised[idx, 0].cpu().numpy()

        # Convert to HU
        hu_min, hu_max = hu_window
        origin_hu_img = denormalize_to_hu(origin_norm, hu_min, hu_max)
        noised_hu_img = denormalize_to_hu(noised_norm, hu_min, hu_max)

        # Apply windowing
        origin_win = apply_window(origin_hu_img, wl=40, ww=400)
        noised_win = apply_window(noised_hu_img, wl=40, ww=400)

        # Rotate
        origin_win = np.rot90(origin_win, k=1)
        noised_win = np.rot90(noised_win, k=1)

        # Compute noise
        origin_hu_std = compute_noise_hu(
            origin[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=True,
            debug=False,
        )
        noised_hu_std = compute_noise_hu(
            noised[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=True,
            debug=False,
        )

        print(f"  [{label}] Origin: {origin_hu_std:.1f} HU → Noised: {noised_hu_std:.1f} HU")

        h, w = origin_win.shape
        aspect = w / h
        fig_height = 8
        fig_width = fig_height * aspect

        # Save origin
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.imshow(origin_win, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{label} Origin - {origin_hu_std:.1f} HU", fontsize=14, pad=10)
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.savefig(origin_dir / f"{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Save noised
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.imshow(noised_win, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{label} Noised - {noised_hu_std:.1f} HU", fontsize=14, pad=10)
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.savefig(noised_dir / f"{label}.png", dpi=150, bbox_inches="tight")
        plt.close()


# -------------------------------------------------------------------------
# Early stopping
# -------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping handler."""

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