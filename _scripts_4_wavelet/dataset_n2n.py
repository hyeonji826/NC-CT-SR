# dataset_n2n.py
# NA-NSN2N Dataset: Noise Augmentation + Neighbor Slice Noise2Noise
# Self-supervised dataset for ultra-low-dose CT enhancement

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, median_filter


class NSN2NDataset(Dataset):
    """
    NS-N2N Dataset with Synthetic Noise Augmentation
    
    Principle: Adjacent CT slices share anatomical structure but have independent noise
    This enables Noise2Noise training without clean ground truth
    
    Key Features:
    - Neighbor slice pairing for noise-independent targets
    - Matched region identification via low-pass filtering
    - Rician noise augmentation for training robustness
    - Adaptive noise estimation from tissue regions
    
    All parameters are configurable via config.yaml
    """

    def __init__(
        self,
        nc_ct_dir: str,
        hu_window: Tuple[float, float] = (-160.0, 240.0),
        patch_size: int = 128,
        min_body_fraction: float = 0.08,
        lpf_sigma: float = 1.0,
        lpf_median_size: int = 3,
        match_threshold: float = 0.005,
        noise_aug_ratio: float = 0.15,
        body_hu_range: Tuple[float, float] = (-500.0, 500.0),
        noise_roi_margin_ratio: float = 0.25,
        noise_tissue_range: Tuple[float, float] = (0.2, 0.8),
        noise_default_std: float = 0.1,
        mode: str = "train",
    ) -> None:
        super().__init__()

        self.nc_ct_dir = Path(nc_ct_dir)
        self.hu_min, self.hu_max = float(hu_window[0]), float(hu_window[1])
        self.patch_size = int(patch_size)
        self.min_body_fraction = float(min_body_fraction)
        self.lpf_sigma = float(lpf_sigma)
        self.lpf_median_size = int(lpf_median_size)
        self.match_threshold = float(match_threshold)
        self.noise_aug_ratio = float(noise_aug_ratio)
        self.mode = mode

        self.body_hu_min, self.body_hu_max = float(body_hu_range[0]), float(body_hu_range[1])

        self.noise_roi_margin_ratio = float(noise_roi_margin_ratio)
        self.noise_tissue_min, self.noise_tissue_max = float(noise_tissue_range[0]), float(noise_tissue_range[1])
        self.noise_default_std = float(noise_default_std)

        files = sorted(list(self.nc_ct_dir.glob("*.nii.gz")) +
                       list(self.nc_ct_dir.glob("*.nii")))
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {self.nc_ct_dir}")

        self.volumes: List[np.ndarray] = []
        self.pairs: List[Tuple[int, int]] = []

        print(f"\n[DATA] Loading NC-CT volumes (mode={mode})...")
        for vol_idx, path in enumerate(files):
            nii = nib.load(str(path))
            vol = nii.get_fdata().astype(np.float32)
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

            if vol.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {vol.shape}")

            H, W, D = vol.shape
            self.volumes.append(vol)

            for z in range(0, D - 1):
                s0 = vol[:, :, z]
                s1 = vol[:, :, z + 1]

                body_mask0 = (s0 > self.body_hu_min) & (s0 < self.body_hu_max)
                body_mask1 = (s1 > self.body_hu_min) & (s1 < self.body_hu_max)
                body_frac = float(body_mask0.sum() + body_mask1.sum()) / (2.0 * H * W)
                if body_frac < self.min_body_fraction:
                    continue

                self.pairs.append((vol_idx, z))

        if not self.pairs:
            raise RuntimeError("No valid slice pairs found.")

        print(f"[DATA] Loaded {len(files)} volumes, {len(self.pairs)} slice pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def _window_and_normalize(self, s: np.ndarray) -> np.ndarray:
        """Apply HU windowing and normalize to [0, 1]"""
        s = np.clip(s, self.hu_min, self.hu_max)
        s = (s - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
        return s.astype(np.float32)

    def _random_crop(self, *arrays):
        """Random spatial crop for data augmentation"""
        if not self.patch_size:
            return arrays

        H, W = arrays[0].shape
        if H <= self.patch_size or W <= self.patch_size:
            return arrays

        top = random.randint(0, H - self.patch_size)
        left = random.randint(0, W - self.patch_size)
        slc = (slice(top, top + self.patch_size),
               slice(left, left + self.patch_size))
        return tuple(arr[slc] for arr in arrays)

    def _compute_weight_map(self, x_i: np.ndarray, x_ip1: np.ndarray) -> np.ndarray:
        """
        Compute matched region weight map W
        
        Regions where low-pass filtered adjacent slices are similar (< threshold)
        indicate matched anatomical structure with independent noise
        """
        H_orig, W_orig = x_i.shape
        base = max(int(getattr(self, "patch_size", min(H_orig, W_orig))), 1)
        scale_h = max(H_orig // base, 1)
        scale_w = max(W_orig // base, 1)
        scale = max(scale_h, scale_w)

        if scale > 1:
            x_i_small = x_i[::scale, ::scale]
            x_ip1_small = x_ip1[::scale, ::scale]

            lp_i_small = median_filter(
                gaussian_filter(x_i_small, sigma=self.lpf_sigma),
                size=self.lpf_median_size,
            )
            lp_ip1_small = median_filter(
                gaussian_filter(x_ip1_small, sigma=self.lpf_sigma),
                size=self.lpf_median_size,
            )

            diff_small = np.abs(lp_i_small - lp_ip1_small)
            W_small = (diff_small <= self.match_threshold).astype(np.float32)

            W_full = np.repeat(np.repeat(W_small, scale, axis=0), scale, axis=1)
            W_full = W_full[:H_orig, :W_orig].astype(np.float32)
            return W_full
        else:
            lp_i = median_filter(
                gaussian_filter(x_i, sigma=self.lpf_sigma),
                size=self.lpf_median_size,
            )
            lp_ip1 = median_filter(
                gaussian_filter(x_ip1, sigma=self.lpf_sigma),
                size=self.lpf_median_size,
            )
            diff = np.abs(lp_i - lp_ip1)
            W = (diff <= self.match_threshold).astype(np.float32)
            return W

    def _estimate_noise_std(self, x: np.ndarray) -> float:
        """
        Estimate noise level from tissue region (normalized 0-1 scale)
        
        Uses central ROI to avoid edge artifacts
        Tissue range ensures measurement from soft tissue, not air or bone
        """
        H, W = x.shape
        m = self.noise_roi_margin_ratio
        h_margin = int(H * m)
        w_margin = int(W * m)
        roi = x[h_margin:H - h_margin, w_margin:W - w_margin]

        t_min, t_max = self.noise_tissue_min, self.noise_tissue_max
        tissue_mask = (roi > t_min) & (roi < t_max)
        if tissue_mask.sum() < 100:
            return float(self.noise_default_std)

        return float(roi[tissue_mask].std())

    def _generate_rician_noise(self, shape: Tuple[int, int], sigma: float) -> np.ndarray:
        """
        Generate Rician noise appropriate for CT imaging
        
        CT reconstruction produces Rician-distributed noise in the magnitude domain
        Bias correction removes the mean shift inherent to Rician distribution
        """
        real = np.random.normal(0, sigma, shape)
        imag = np.random.normal(0, sigma, shape)
        magnitude = np.sqrt(real**2 + imag**2)

        bias = sigma * np.sqrt(np.pi / 2)
        noise = magnitude - bias

        return noise.astype(np.float32)

    def __getitem__(self, idx: int):
        """
        Return matched slice pair with synthetic noise augmentation
        
        Returns dict with:
            x_i: Original noisy slice i
            x_i_aug: Slice i with additional synthetic noise (training only)
            x_ip1: Adjacent noisy slice i+1
            x_mid: Average of x_i and x_ip1 (compatibility)
            W: Matched region weight map
            noise_synthetic: Added noise map (for analysis)
        """
        vol_idx, z = self.pairs[idx]
        vol = self.volumes[vol_idx]

        s0 = vol[:, :, z]
        s1 = vol[:, :, z + 1]

        x_i = self._window_and_normalize(s0)
        x_ip1 = self._window_and_normalize(s1)
        x_mid = 0.5 * (x_i + x_ip1)

        W = self._compute_weight_map(x_i, x_ip1)

        if self.mode == "train":
            noise_std = self._estimate_noise_std(x_i)
            aug_sigma = noise_std * self.noise_aug_ratio
            noise_synthetic = self._generate_rician_noise(x_i.shape, aug_sigma)
            x_i_aug = np.clip(x_i + noise_synthetic, 0.0, 1.0)
        else:
            noise_synthetic = np.zeros_like(x_i)
            x_i_aug = x_i

        x_i, x_i_aug, x_ip1, x_mid, W, noise_synthetic = self._random_crop(
            x_i, x_i_aug, x_ip1, x_mid, W, noise_synthetic
        )

        x_i_t = torch.from_numpy(x_i).unsqueeze(0)
        x_i_aug_t = torch.from_numpy(x_i_aug).unsqueeze(0)
        x_ip1_t = torch.from_numpy(x_ip1).unsqueeze(0)
        x_mid_t = torch.from_numpy(x_mid).unsqueeze(0)
        W_t = torch.from_numpy(W).unsqueeze(0)
        noise_syn_t = torch.from_numpy(noise_synthetic).unsqueeze(0)

        return {
            "x_i": x_i_t,
            "x_i_aug": x_i_aug_t,
            "x_ip1": x_ip1_t,
            "x_mid": x_mid_t,
            "W": W_t,
            "noise_synthetic": noise_syn_t,
        }