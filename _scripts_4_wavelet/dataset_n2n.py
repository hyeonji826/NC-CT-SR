# E:\LD-CT SR\_scripts_4_wavelet\dataset_n2n.py
# NS-N2N Dataset: neighboring-slice pairs + weight map
#
# 출력:
#   x_i    : [1,H,W]   (slice z,  0~1 정규화)
#   x_ip1  : [1,H,W]   (slice z+1,0~1 정규화)
#   x_mid  : [1,H,W]   (0.5*(x_i+x_ip1))
#   W      : [1,H,W]   (matched 영역 binary weight map)

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
    def __init__(
        self,
        nc_ct_dir: str,
        hu_window: Tuple[float, float] = (-160.0, 240.0),
        patch_size: int = 128,
        min_body_fraction: float = 0.05,
        lpf_sigma: float = 1.0,
        lpf_median_size: int = 3,
        match_threshold: float = 0.01,
        mode: str = "train",
    ) -> None:
        super().__init__()
        self.nc_ct_dir = Path(nc_ct_dir)
        self.hu_min, self.hu_max = float(hu_window[0]), float(hu_window[1])
        self.patch_size = patch_size
        self.min_body_fraction = float(min_body_fraction)
        self.lpf_sigma = float(lpf_sigma)
        self.lpf_median_size = int(lpf_median_size)
        self.match_threshold = float(match_threshold)
        self.mode = mode

        files = sorted(list(self.nc_ct_dir.glob("*.nii.gz")) +
                       list(self.nc_ct_dir.glob("*.nii")))
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {self.nc_ct_dir}")

        self.volumes: List[np.ndarray] = []
        self.pairs: List[Tuple[int, int]] = []  # (vol_idx, z) for pair (z, z+1)

        print("\n📂 Loading NC-CT volumes for NS-N2N dataset ...")
        for vol_idx, path in enumerate(files):
            nii = nib.load(str(path))
            vol = nii.get_fdata().astype(np.float32)
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

            if vol.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {vol.shape} for {path}")

            H, W, D = vol.shape
            self.volumes.append(vol)

            # (z,z+1) 쌍 만들기
            for z in range(0, D - 1):
                s0 = vol[:, :, z]
                s1 = vol[:, :, z + 1]

                body_mask0 = (s0 > -500) & (s0 < 500)
                body_mask1 = (s1 > -500) & (s1 < 500)
                body_frac = float(body_mask0.sum() + body_mask1.sum()) / (2.0 * H * W)
                if body_frac < self.min_body_fraction:
                    continue

                self.pairs.append((vol_idx, z))

        if not self.pairs:
            raise RuntimeError("No valid slice pairs found for NS-N2N dataset.")

        print(f"   Loaded {len(files)} volumes, {len(self.pairs)} usable pairs (mode={self.mode})")

    def __len__(self) -> int:
        return len(self.pairs)

    def _window_and_normalize(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(s, self.hu_min, self.hu_max)
        s = (s - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
        return s.astype(np.float32)

    def _random_crop(self, a0: np.ndarray, a1: np.ndarray,
                     a2: np.ndarray, w: np.ndarray):
        if not self.patch_size:
            return a0, a1, a2, w

        H, W = a0.shape
        if H <= self.patch_size or W <= self.patch_size:
            return a0, a1, a2, w

        top = random.randint(0, H - self.patch_size)
        left = random.randint(0, W - self.patch_size)
        slc = (slice(top, top + self.patch_size),
               slice(left, left + self.patch_size))
        return a0[slc], a1[slc], a2[slc], w[slc]

    def _compute_weight_map(self, x_i: np.ndarray, x_ip1: np.ndarray) -> np.ndarray:
        """
        Low-pass 필터 후, 차이가 작은 영역을 matched 영역으로 사용.
        슬라이스가 클 때는 downsample 해서 필터 연산량 줄인 뒤
        다시 upsample 해서 원래 해상도로 되돌린다.
        """
        H_orig, W_orig = x_i.shape  # 원래 크기 저장

        # patch_size 기준으로 downsample factor 결정
        # 예: H=W=512, patch_size=128 이면 scale=4 -> 128x128에서만 필터
        base = max(int(getattr(self, "patch_size", min(H_orig, W_orig))), 1)
        scale_h = max(H_orig // base, 1)
        scale_w = max(W_orig // base, 1)
        scale = max(scale_h, scale_w)

        if scale > 1:
            # downsample
            x_i_small   = x_i[::scale, ::scale]
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

            # upsample back to original size
            W_full = np.repeat(np.repeat(W_small, scale, axis=0), scale, axis=1)
            W_full = W_full[:H_orig, :W_orig].astype(np.float32)
            return W_full
        else:
            # 슬라이스가 이미 작으면 기존 방식 그대로
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


    def __getitem__(self, idx: int):
        vol_idx, z = self.pairs[idx]
        vol = self.volumes[vol_idx]

        s0 = vol[:, :, z]
        s1 = vol[:, :, z + 1]

        x_i = self._window_and_normalize(s0)
        x_ip1 = self._window_and_normalize(s1)
        x_mid = 0.5 * (x_i + x_ip1)

        W = self._compute_weight_map(x_i, x_ip1)

        x_i, x_ip1, x_mid, W = self._random_crop(x_i, x_ip1, x_mid, W)

        x_i_t = torch.from_numpy(x_i).unsqueeze(0)      # [1,H,W]
        x_ip1_t = torch.from_numpy(x_ip1).unsqueeze(0)
        x_mid_t = torch.from_numpy(x_mid).unsqueeze(0)
        W_t = torch.from_numpy(W).unsqueeze(0)

        return {
            "x_i": x_i_t,
            "x_ip1": x_ip1_t,
            "x_mid": x_mid_t,
            "W": W_t,
        }
