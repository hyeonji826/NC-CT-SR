# dataset_n2n.py
# NS-N2N Dataset: Neighbor-Slice Noise2Noise + CT-like Noise Augmentation
# Self-supervised ultra-low-dose CT denoising with 3D input

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
    NS-N2N Dataset with CT-like Synthetic Noise Augmentation

    Principle:
        - Adjacent CT slices share anatomy but have largely independent noise
        - Use slice z as input, slice z±1 as noise-independent target (Noise2Noise)
        - Optionally add realistic synthetic CT noise to the input only

    Model interface (train_n2n.py / losses_n2n.py와 호환):
        __getitem__ returns dict with:
            x_i        : (1, H, W)  center slice (원본 NC-CT, 정규화)
            x_i_aug    : (1, 5, H, W)  5-slice volume, center에만 synthetic noise 적용
            x_ip1      : (1, H, W)  neighbor slice (target 역할)
            x_mid      : (1, H, W)  x_i, x_ip1의 평균(호환용, loss에서 현재 사용 X)
            W          : (1, H, W)  matched region mask (0~1)
            noise_synthetic : (1, H, W)  center에 추가한 synthetic noise (정규화 공간)
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
        noise_aug_ratio: float = 0.18,
        body_hu_range: Tuple[float, float] = (-500.0, 500.0),
        noise_roi_margin_ratio: float = 0.18,   # (현재 버전에서는 사용하지 않지만, config와 인터페이스 맞추기용)
        noise_tissue_range: Tuple[float, float] = (0.25, 0.80),  # same
        noise_default_std: float = 0.1,        # same
        mode: str = "train",
    ) -> None:
        super().__init__()

        # 기본 설정
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

        # NIfTI 파일 로드
        files = sorted(list(self.nc_ct_dir.glob("*.nii.gz")) +
                       list(self.nc_ct_dir.glob("*.nii")))
        if not files:
            raise FileNotFoundError(f"No NIfTI files found in {self.nc_ct_dir}")

        self.volumes: List[np.ndarray] = []
        self.pairs: List[Tuple[int, int]] = []   # (volume_index, z_index)

        for vol_idx, path in enumerate(files):
            img = nib.load(str(path))
            vol = img.get_fdata().astype(np.float32)

            # 4D면 마지막 채널 0번만 사용 (보통 (H,W,Z,1) 형태)
            if vol.ndim == 4:
                vol = vol[..., 0]

            if vol.ndim != 3:
                raise ValueError(f"Volume {path} has invalid shape {vol.shape}, expected 3D.")

            H, W, D = vol.shape
            self.volumes.append(vol)

            # slice 별 body fraction 확인해서 쓸만한 center slice만 pairs에 넣는다
            body_fracs = []
            for z in range(D):
                s = vol[:, :, z]
                body_mask = self._make_body_mask(s)
                body_fracs.append(float(body_mask.mean()))

            for z in range(D):
                if body_fracs[z] < self.min_body_fraction:
                    continue

                # 이 slice를 center로 쓸 때, 이웃 slice 중 최소 1개는 body_fraction 충분해야 함
                has_neighbor = False
                if z > 0 and body_fracs[z - 1] >= self.min_body_fraction:
                    has_neighbor = True
                if z < D - 1 and body_fracs[z + 1] >= self.min_body_fraction:
                    has_neighbor = True

                if not has_neighbor:
                    continue

                self.pairs.append((vol_idx, z))

        if not self.pairs:
            raise RuntimeError("No valid slice pairs found for NS-N2N.")

        print(f"[DATA] Loaded {len(files)} volumes, {len(self.pairs)} slice pairs")

    # ------------------------------------------------------------------
    # 유틸 함수들
    # ------------------------------------------------------------------
    def _window_and_normalize(self, s: np.ndarray) -> np.ndarray:
        """HU window 적용 후 [0,1] 정규화"""
        s = np.clip(s, self.hu_min, self.hu_max)
        s = (s - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
        return s.astype(np.float32)

    def _make_body_mask(self, hu: np.ndarray) -> np.ndarray:
        """몸통 영역 mask (HU 기반)"""
        mask = (hu >= self.body_hu_min) & (hu <= self.body_hu_max)
        return mask.astype(np.float32)

    def _compute_match_map(self, x_i: np.ndarray, x_ip1: np.ndarray, body_mask: np.ndarray) -> np.ndarray:
        """
        Neighbor slice 간 low-pass diff 기반 matched region mask W 계산.
        x_i, x_ip1는 [0,1] 정규화된 이미지.
        """
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
        # body 영역 밖은 신뢰하지 않음
        W *= body_mask.astype(np.float32)
        return W

    def _random_crop_5(
        self,
        vol5: np.ndarray,
        *arrays: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """
        5-slice volume(5,H,W)과 여러 2D 배열(H,W)에 동일한 random crop 적용
        """
        if not self.patch_size:
            return (vol5, *arrays)

        _, H, W = vol5.shape
        if H <= self.patch_size or W <= self.patch_size:
            return (vol5, *arrays)

        top = random.randint(0, H - self.patch_size)
        left = random.randint(0, W - self.patch_size)
        slc_h = slice(top, top + self.patch_size)
        slc_w = slice(left, left + self.patch_size)

        vol5_c = vol5[:, slc_h, slc_w]
        cropped = [a[slc_h, slc_w] for a in arrays]
        return (vol5_c, *cropped)

    def _maybe_flip(
        self,
        vol5: np.ndarray,
        *arrays: np.ndarray,
    ) -> Tuple[np.ndarray, ...]:
        """
        간단한 좌우/상하 flip augmentation (train 모드에서만 호출)
        """
        if self.mode != "train":
            return (vol5, *arrays)

        # 상하 flip
        if random.random() < 0.5:
            vol5 = np.flip(vol5, axis=1)   # H
            arrays = tuple(np.flip(a, axis=0) for a in arrays)

        # 좌우 flip
        if random.random() < 0.5:
            vol5 = np.flip(vol5, axis=2)   # W
            arrays = tuple(np.flip(a, axis=1) for a in arrays)

        return (vol5, *arrays)

    def _add_ct_like_noise(self, hu: np.ndarray, scale: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        실제 NC-CT의 NPS/통계를 흉내내는 synthetic CT-like noise 추가.
        - 중/고주파 correlated Gaussian noise
        - 저주파 shading
        - 수직/수평 streak artifact

        Args:
            hu: HU 공간의 입력 이미지
            scale: 노이즈 배율 (1.0 = 원본 수준, 1.5 = 1.5배 noisy)
        
        입력/출력은 HU 공간.
        """
        hu = hu.astype(np.float32)
        H, W = hu.shape

        body = self._make_body_mask(hu)
        body_vals = hu[body > 0.5]
        if body_vals.size < 100:
            noise = np.random.normal(0.0, 5.0 * scale, size=hu.shape).astype(np.float32)
            return hu + noise, noise

        sigma_origin = float(body_vals.std())

        # scale 파라미터를 사용하여 노이즈 강도 조절
        # scale=1.0이면 원본과 비슷한 수준, scale=1.5면 1.5배 noisy
        target_mult = np.random.uniform(1.2, 1.6) * scale
        target_sigma = sigma_origin * target_mult

        # 1) correlated Gaussian noise (중/고주파)
        g = np.random.normal(0.0, 1.0, size=hu.shape).astype(np.float32)
        g = gaussian_filter(g, sigma=np.random.uniform(0.7, 1.5))

        # 2) 저주파 shading field
        lf = np.random.normal(0.0, 1.0, size=hu.shape).astype(np.float32)
        lf = gaussian_filter(lf, sigma=np.random.uniform(20.0, 40.0))

        # 3) 수직/수평 streak
        yy, xx = np.mgrid[0:H, 0:W]
        streak = np.zeros_like(hu, dtype=np.float32)

        if np.random.rand() < 0.8:
            amp = np.random.uniform(2.0, 6.0)
            freq = np.random.uniform(0.01, 0.03)
            phase = np.random.uniform(0.0, 2.0 * np.pi)
            streak += amp * np.sin(2.0 * np.pi * freq * yy + phase)

        if np.random.rand() < 0.8:
            amp = np.random.uniform(2.0, 6.0)
            freq = np.random.uniform(0.01, 0.03)
            phase = np.random.uniform(0.0, 2.0 * np.pi)
            streak += amp * np.sin(2.0 * np.pi * freq * xx + phase)

        noise_raw = 0.6 * g + 0.3 * lf + 0.1 * streak

        body_noise = noise_raw[body > 0.5]
        sigma_raw = float(body_noise.std()) + 1e-6

        if sigma_raw > 0.0 and target_sigma > sigma_origin:
            desired_increase = target_sigma - sigma_origin
            scale = desired_increase / sigma_raw
        else:
            scale = 0.0

        noise = noise_raw * scale
        noisy_hu = hu + noise

        return noisy_hu.astype(np.float32), noise.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """
        Return:
            x_i          : (1, H, W) center slice (원본)
            x_i_aug      : (1, 5, H, W) center에 synthetic noise + neighbor context
            x_ip1        : (1, H, W) neighbor slice
            x_mid        : (1, H, W) (x_i + x_ip1) / 2 (현재 loss에서는 사용 X)
            W            : (1, H, W) matched mask
            noise_synthetic : (1, H, W) center에 추가한 synthetic noise (정규화 공간)
        """
        vol_idx, z = self.pairs[idx]
        vol = self.volumes[vol_idx]
        H, W, D = vol.shape

        # center, neighbor 선택 (z-1 또는 z+1 중 랜덤)
        if z == 0:
            z_pair = 1
        elif z == D - 1:
            z_pair = D - 2
        else:
            z_pair = z - 1 if random.random() < 0.5 else z + 1

        hu_center = vol[:, :, z]
        hu_pair = vol[:, :, z_pair]
        hu_mid = 0.5 * (hu_center + hu_pair)

        # 5-slice context (z-2 ~ z+2, boundary는 clamp)
        indices = []
        for offset in [-2, -1, 0, 1, 2]:
            _z = z + offset
            _z = 0 if _z < 0 else (D - 1 if _z >= D else _z)
            indices.append(_z)

        slices_5 = np.stack([vol[:, :, zz] for zz in indices], axis=0)  # (5, H, W)
        slices_5_norm = np.stack(
            [self._window_and_normalize(s) for s in slices_5],
            axis=0,
        )  # (5, H, W)

        x_center = self._window_and_normalize(hu_center)
        x_pair = self._window_and_normalize(hu_pair)
        x_mid = self._window_and_normalize(hu_mid)

        # synthetic CT-like noise: center slice에만 추가 (입력만 더 noisy)
        # noise_aug_ratio를 스케일로 사용 (1.0 = 원본과 동일, 1.5 = 1.5배 noisy)
        if self.mode == "train":
            noisy_hu, noise_hu = self._add_ct_like_noise(hu_center, scale=self.noise_aug_ratio)
            x_center_noisy = self._window_and_normalize(noisy_hu)
            noise_synthetic = (x_center_noisy - x_center).astype(np.float32)
        else:
            x_center_noisy = x_center.copy()
            noise_synthetic = np.zeros_like(x_center, dtype=np.float32)

        # 5-slice volume에서 center slice만 noisy 버전으로 교체
        x_5slices_aug = slices_5_norm.copy()
        x_5slices_aug[2] = x_center_noisy  # center index = 2

        # matched regions mask W (body mask 포함)
        body_mask = self._make_body_mask(hu_center)
        W = self._compute_match_map(x_center, x_pair, body_mask)

        # random crop (5-volume + 2D maps)
        x_5slices_aug, x_center, x_pair, x_mid, W, noise_synthetic = self._random_crop_5(
            x_5slices_aug, x_center, x_pair, x_mid, W, noise_synthetic
        )

        # flip augmentation
        x_5slices_aug, x_center, x_pair, x_mid, W, noise_synthetic = self._maybe_flip(
            x_5slices_aug, x_center, x_pair, x_mid, W, noise_synthetic
        )

        # np.flip 때문에 stride가 음수가 될 수 있으니, 모두 contiguous 로 복사
        x_5slices_aug = np.ascontiguousarray(x_5slices_aug)
        x_center = np.ascontiguousarray(x_center)
        x_pair = np.ascontiguousarray(x_pair)
        x_mid = np.ascontiguousarray(x_mid)
        W = np.ascontiguousarray(W)
        noise_synthetic = np.ascontiguousarray(noise_synthetic)

        # torch tensor 변환
        x_5slices_aug_t = torch.from_numpy(x_5slices_aug).unsqueeze(0)  # (1,5,H,W)
        x_center_t = torch.from_numpy(x_center).unsqueeze(0)            # (1,H,W)
        x_pair_t = torch.from_numpy(x_pair).unsqueeze(0)
        x_mid_t = torch.from_numpy(x_mid).unsqueeze(0)
        W_t = torch.from_numpy(W).unsqueeze(0)
        noise_syn_t = torch.from_numpy(noise_synthetic).unsqueeze(0)


        return {
            "x_i": x_center_t,
            "x_i_aug": x_5slices_aug_t,
            "x_ip1": x_pair_t,
            "x_mid": x_mid_t,
            "W": W_t,
            "noise_synthetic": noise_syn_t,
        }