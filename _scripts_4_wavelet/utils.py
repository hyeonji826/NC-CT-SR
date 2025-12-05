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
    use_highpass: bool = False,
    debug: bool = False,
) -> float:
    hu_min, hu_max = hu_window
    arr = x_01[0, 0].detach().cpu().numpy()
    H, W = arr.shape

    # 중앙 ROI
    h_margin = int(H * (1.0 - roi_h) / 2.0)
    w_margin = int(W * (1.0 - roi_w) / 2.0)
    roi = arr[h_margin : H - h_margin, w_margin : W - w_margin]

    # Normalized 공간에서 body mask (air/background 제외)
    body_mask_norm = (roi > 0.15) & (roi < 0.85)
    if body_mask_norm.sum() < 100:
        return 0.0

    # body 영역만 HU로 변환
    roi_hu = roi * (hu_max - hu_min) + hu_min

    # Raw std (의미 해석이 제일 쉬움)
    raw_std_hu = float(roi_hu[body_mask_norm].std())

    if use_highpass:
        lp = gaussian_filter(roi_hu, sigma=1.0)
        hp = roi_hu - lp
        hp_std_hu = float(hp[body_mask_norm].std())
        noise_std_hu = hp_std_hu

        if debug:
            ratio = hp_std_hu / raw_std_hu if raw_std_hu > 0 else 0.0
            print(
                f"    [DEBUG] Raw std: {raw_std_hu:.1f} HU | "
                f"HP std: {hp_std_hu:.1f} HU | Ratio: {ratio:.2%}"
            )
    else:
        noise_std_hu = raw_std_hu
        if debug:
            print(f"    [DEBUG] Raw std: {raw_std_hu:.1f} HU (no HP filtering)")

    return noise_std_hu


# -------------------------------------------------------------------------
# Sample saving
# -------------------------------------------------------------------------

def save_simple_samples(
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

        noisy_np = noisy[idx, 0].cpu().numpy()
        denoised_np = denoised[idx, 0].detach().cpu().numpy()

        # 보기 좋게 회전
        noisy_np = np.rot90(noisy_np, k=1)
        denoised_np = np.rot90(denoised_np, k=1)

        # HU 기반 noise 계산
        noisy_hu = compute_noise_hu(
            noisy[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=False,
            debug=True,
        )
        denoised_hu = compute_noise_hu(
            denoised[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=False,
            debug=True,
        )

        reduction = (
            (noisy_hu - denoised_hu) / noisy_hu * 100 if noisy_hu > 0 else 0.0
        )
        print(
            f"  [{label}] Original: {noisy_hu:.1f} HU → "
            f"Denoised: {denoised_hu:.1f} HU "
            f"({reduction:.1f}% reduction) | Shape: {noisy_np.shape}"
        )

        # figure 크기
        h, w = noisy_np.shape
        aspect_ratio = w / h
        fig_height = 8
        fig_width = fig_height * aspect_ratio

        # 라벨별 하위 폴더 (HN / LN)
        label_dir = denoise_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        # denoised 이미지만 저장 (epoch별)
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.imshow(denoised_np, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(
            f"{label} Denoised - {denoised_hu:.1f} HU ({reduction:.1f}% ↓)",
            fontsize=14,
            pad=10,
        )
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        denoise_path = label_dir / f"epoch_{epoch:03d}.png"
        plt.savefig(denoise_path, dpi=150, bbox_inches="tight")
        plt.close()


def save_origin_noised_samples(
    origin: torch.Tensor,      # (B,1,H,W) clean (NC)
    noised: torch.Tensor,      # (B,1,H,W) synthetic LD
    origin_dir: Path,
    noised_dir: Path,
    hu_window: Tuple[float, float],
    body_hu_range: Tuple[float, float],
):
    """
    Origin(깨끗한 NC)과 synthetic noised 이미지를 한 번만 저장하는 함수.

    Folder 구조:
        origin/
          HN.png
          LN.png
        noised/
          HN.png
          LN.png
    """
    origin_dir.mkdir(parents=True, exist_ok=True)
    noised_dir.mkdir(parents=True, exist_ok=True)

    labels = ["HN", "LN"]
    for idx in range(min(2, origin.shape[0])):
        label = labels[idx]

        origin_np = np.rot90(origin[idx, 0].cpu().numpy(), k=1)
        noised_np = np.rot90(noised[idx, 0].cpu().numpy(), k=1)

        origin_hu = compute_noise_hu(
            origin[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=False,
            debug=True,
        )
        noised_hu = compute_noise_hu(
            noised[idx : idx + 1],
            hu_window,
            body_hu_range,
            use_highpass=False,
            debug=True,
        )

        print(f"  [{label}] Clean: {origin_hu:.1f} HU → Noised: {noised_hu:.1f} HU")

        h, w = origin_np.shape
        aspect = w / h
        fig_height = 8
        fig_width = fig_height * aspect

        # origin
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.imshow(origin_np, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{label} Origin - {origin_hu:.1f} HU", fontsize=14, pad=10)
        plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        plt.savefig(origin_dir / f"{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # noised
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        ax.imshow(noised_np, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{label} Noised - {noised_hu:.1f} HU", fontsize=14, pad=10)
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