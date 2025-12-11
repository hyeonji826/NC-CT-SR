# dataset_n2n.py
# NS-N2N Dataset: Neighbor-Slice Noise2Noise + NPS-guided CT-like Noise Augmentation
# Self-supervised ultra-low-dose CT denoising with 3D input

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter, median_filter


class NSN2NDataset(Dataset):
    def __init__(
        self,
        nc_ct_dir: str,
        hu_window: Tuple[float, float] = (-160.0, 240.0),
        patch_size: int = 128,
        min_body_fraction: float = 0.08,
        lpf_sigma: float = 1.0,
        lpf_median_size: int = 3,
        match_threshold: float = 0.005,
        noise_aug_ratio: float = 1.5,  # NPS 기반 noise amplification factor
        body_hu_range: Tuple[float, float] = (-500.0, 500.0),
        noise_roi_margin_ratio: float = 0.18,
        noise_tissue_range: Tuple[float, float] = (0.25, 0.80),
        noise_default_std: float = 0.1,
        mode: str = "train",
        slice_noise_csv: str | None = None,
        augment_streaks: bool = False,  # Stage 2에서 True
        streak_strength: float = 0.15,  # Streak 강도 (0~1)
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
        self.augment_streaks = augment_streaks
        self.streak_strength = streak_strength
        self.mode = mode

        self.body_hu_min, self.body_hu_max = float(body_hu_range[0]), float(body_hu_range[1])

        # NPS 기반 slice-level noise 표 (optional, per patient,z)
        self.slice_noise_map: dict[tuple[str, int], float] = {}
        self.slice_noise_mean: float | None = None
        if slice_noise_csv is not None:
            csv_path = Path(slice_noise_csv)
            if not csv_path.exists():
                print(f"[WARN] slice_noise_csv not found: {csv_path} (adaptive NPS noise OFF)")
            else:
                df_noise = pd.read_csv(csv_path)
                if not {"patient", "z", "noise_std"}.issubset(df_noise.columns):
                    print(
                        f"[WARN] slice_noise_csv columns invalid: "
                        f"{df_noise.columns.tolist()} (expected patient,z,noise_std)"
                    )
                else:
                    df_noise["patient"] = df_noise["patient"].astype(str)
                    self.slice_noise_mean = float(df_noise["noise_std"].mean())
                    self.slice_noise_map = {
                        (str(row.patient), int(row.z)): float(row.noise_std)
                        for _, row in df_noise.iterrows()
                    }
                    print(
                        f"[INFO] Loaded slice_noise_csv from {csv_path} "
                        f"(mean noise_std={self.slice_noise_mean:.2f}, "
                        f"entries={len(self.slice_noise_map)})"
                    )

        # 볼륨 인덱스 → patient ID 매핑 (ex. '25980_0000.nii.gz' → '25980')
        self.volume_patient_ids: list[str] = []

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

            # NIfTI 파일명에서 patient ID 추출 (예: '25980_0000.nii.gz' → '25980')
            stem = path.stem  # '25980_0000'
            patient_id = stem.split("_")[0]
            self.volume_patient_ids.append(str(patient_id))

            # 4D면 마지막 채널 0번만 사용
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
                # neighbor가 존재하는지도 확인
                has_neighbor = (z > 0 and body_fracs[z - 1] > self.min_body_fraction) or \
                               (z < D - 1 and body_fracs[z + 1] > self.min_body_fraction)
                if has_neighbor:
                    self.pairs.append((vol_idx, z))

        print(f"[NSN2NDataset] Loaded {len(self.volumes)} volumes, {len(self.pairs)} valid pairs")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _make_body_mask(self, hu: np.ndarray) -> np.ndarray:
        """Body mask (HU 공간): air 제외, bone 제외"""
        mask = (hu > self.body_hu_min) & (hu < self.body_hu_max)
        return mask.astype(np.float32)

    def _window_and_normalize(self, hu: np.ndarray) -> np.ndarray:
        """HU → [0,1] 정규화"""
        hu = np.clip(hu, self.hu_min, self.hu_max)
        norm = (hu - self.hu_min) / (self.hu_max - self.hu_min + 1e-7)
        return norm.astype(np.float32)

    def _compute_match_map(
        self,
        x_i: np.ndarray,
        x_ip1: np.ndarray,
        body_mask: np.ndarray,
    ) -> np.ndarray:
        """
        matching mask W: low-pass correlation이 높은 영역 (flat regions)
        """
        lpf_i = median_filter(x_i, size=self.lpf_median_size)
        lpf_i = gaussian_filter(lpf_i, sigma=self.lpf_sigma)

        lpf_ip1 = median_filter(x_ip1, size=self.lpf_median_size)
        lpf_ip1 = gaussian_filter(lpf_ip1, sigma=self.lpf_sigma)

        diff = np.abs(lpf_i - lpf_ip1)
        W = ((diff < self.match_threshold) & (body_mask > 0.5)).astype(np.float32)
        return W

    def _random_crop_5(
        self,
        x_5slices: np.ndarray,
        x_i: np.ndarray,
        x_ip1: np.ndarray,
        x_mid: np.ndarray,
        W: np.ndarray,
        noise_synthetic: np.ndarray,
    ):
        """5-slice volume과 2D maps를 동시에 crop"""
        _, H, W_dim = x_5slices.shape
        h, w = self.patch_size, self.patch_size
        
        # ★ patch_size=0이면 crop 안 하고 전체 이미지 반환
        if self.patch_size == 0:
            return x_5slices, x_i, x_ip1, x_mid, W, noise_synthetic

        if H < h or W_dim < w:
            # 패치 크기보다 작으면 zero-padding
            x_5slices_pad = np.zeros((5, h, w), dtype=np.float32)
            x_i_pad = np.zeros((h, w), dtype=np.float32)
            x_ip1_pad = np.zeros((h, w), dtype=np.float32)
            x_mid_pad = np.zeros((h, w), dtype=np.float32)
            W_pad = np.zeros((h, w), dtype=np.float32)
            noise_pad = np.zeros((h, w), dtype=np.float32)

            # 중앙에 배치
            oh = (h - H) // 2
            ow = (w - W_dim) // 2
            x_5slices_pad[:, oh:oh+H, ow:ow+W_dim] = x_5slices
            x_i_pad[oh:oh+H, ow:ow+W_dim] = x_i
            x_ip1_pad[oh:oh+H, ow:ow+W_dim] = x_ip1
            x_mid_pad[oh:oh+H, ow:ow+W_dim] = x_mid
            W_pad[oh:oh+H, ow:ow+W_dim] = W
            noise_pad[oh:oh+H, ow:ow+W_dim] = noise_synthetic

            return x_5slices_pad, x_i_pad, x_ip1_pad, x_mid_pad, W_pad, noise_pad

        # Random crop
        top = random.randint(0, H - h)
        left = random.randint(0, W_dim - w)

        return (
            x_5slices[:, top:top+h, left:left+w],
            x_i[top:top+h, left:left+w],
            x_ip1[top:top+h, left:left+w],
            x_mid[top:top+h, left:left+w],
            W[top:top+h, left:left+w],
            noise_synthetic[top:top+h, left:left+w],
        )

    def _maybe_flip(
        self,
        x_5slices: np.ndarray,
        x_i: np.ndarray,
        x_ip1: np.ndarray,
        x_mid: np.ndarray,
        W: np.ndarray,
        noise_synthetic: np.ndarray,
    ):
        """Random flip augmentation"""
        if random.random() < 0.5:
            # horizontal flip
            x_5slices = np.flip(x_5slices, axis=2)
            x_i = np.flip(x_i, axis=1)
            x_ip1 = np.flip(x_ip1, axis=1)
            x_mid = np.flip(x_mid, axis=1)
            W = np.flip(W, axis=1)
            noise_synthetic = np.flip(noise_synthetic, axis=1)

        if random.random() < 0.5:
            # vertical flip
            x_5slices = np.flip(x_5slices, axis=1)
            x_i = np.flip(x_i, axis=0)
            x_ip1 = np.flip(x_ip1, axis=0)
            x_mid = np.flip(x_mid, axis=0)
            W = np.flip(W, axis=0)
            noise_synthetic = np.flip(noise_synthetic, axis=0)

        return x_5slices, x_i, x_ip1, x_mid, W, noise_synthetic

    # ------------------------------------------------------------------
    # ★ NPS-GUIDED SYNTHETIC NOISE GENERATION ★
    # ------------------------------------------------------------------
    def _add_ct_like_noise_nps_guided(
        self, 
        hu: np.ndarray, 
        scale: float = 1.5,
        augment_streaks: bool = False,
        streak_strength: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:

        hu = hu.astype(np.float32)
        H, W = hu.shape

        # Body mask로 tissue 영역만 분석
        body = self._make_body_mask(hu)
        body_vals = hu[body > 0.5]
        
        if body_vals.size < 100:
            # Body가 너무 작으면 단순 Gaussian
            noise = np.random.normal(0.0, 5.0 * scale, size=hu.shape).astype(np.float32)
            return hu + noise, noise

        # 현재 이미지의 noise statistics 추정
        sigma_origin = float(body_vals.std())
        
        # Target noise level 결정 (adaptive)
        # scale 파라미터로 전체 증폭 조절
        target_mult = np.random.uniform(1.1, 1.4) * scale
        target_sigma = sigma_origin * target_mult

        # ===== COMPONENT 1: Low-frequency shading (NPS 기반) =====
        # NPS 분석: low-freq가 수백~수천 배 강함
        # Adaptive sigma: image std에 비례
        lf_sigma = np.random.uniform(25.0, 45.0)  # 기존보다 더 넓은 범위
        lf = np.random.normal(0.0, 1.0, size=hu.shape).astype(np.float32)
        lf = gaussian_filter(lf, sigma=lf_sigma)
        
        # ===== COMPONENT 2: Mid-frequency correlated noise =====
        # 기존 correlated Gaussian을 약화
        mf_sigma = np.random.uniform(0.8, 1.2)
        mf = np.random.normal(0.0, 1.0, size=hu.shape).astype(np.float32)
        mf = gaussian_filter(mf, sigma=mf_sigma)
        
        # ===== COMPONENT 3: Horizontal-dominant streak (NPS 기반) =====
        # NPS 분석: Horizontal이 Vertical보다 2~3배 강함
        yy, xx = np.mgrid[0:H, 0:W]
        streak = np.zeros_like(hu, dtype=np.float32)
        
        # Horizontal streak (더 강하고 높은 확률)
        if np.random.rand() < 0.85:  # 85% 확률
            amp_h = np.random.uniform(3.0, 7.0)  # 더 강한 amplitude
            freq_h = np.random.uniform(0.012, 0.035)
            phase_h = np.random.uniform(0.0, 2.0 * np.pi)
            streak += amp_h * np.sin(2.0 * np.pi * freq_h * yy + phase_h)
        
        # Vertical streak (약하고 낮은 확률)
        if np.random.rand() < 0.35:  # 35% 확률 (H:V ≈ 2.4:1)
            amp_v = np.random.uniform(1.5, 3.5)  # 더 약한 amplitude
            freq_v = np.random.uniform(0.012, 0.035)
            phase_v = np.random.uniform(0.0, 2.0 * np.pi)
            streak += amp_v * np.sin(2.0 * np.pi * freq_v * xx + phase_v)
        
        # ===== WEIGHTED COMBINATION (NPS 기반) =====
        # augment_streaks=True: Stage별로 streak 포함 (artifact 학습)
        # augment_streaks=False: Random noise만 (streak 제외)
        if augment_streaks:
            # Stage 1: 약한 streak (모델이 artifact 패턴 학습)
            # Stage 2: 더 강한 streak (artifact 제거 강화)
            noise_raw = (1.0 - streak_strength) * (0.4 * mf + 0.6 * lf) + streak_strength * streak
        else:
            # Streak 완전 제외 (랜덤 noise만)
            noise_raw = 0.4 * mf + 0.6 * lf
        
        # Body 영역에서 noise statistics 측정
        body_noise = noise_raw[body > 0.5]
        sigma_raw = float(body_noise.std()) + 1e-6
        
        # Target sigma에 맞춰 scaling
        if sigma_raw > 0.0 and target_sigma > sigma_origin:
            desired_increase = np.sqrt(target_sigma**2 - sigma_origin**2)
            scale_factor = desired_increase / sigma_raw
        else:
            scale_factor = 0.0
        
        noise = noise_raw * scale_factor
        
        # Body mask 적용: body 외부는 noise 0
        noise = noise * body

        noisy_hu = hu + noise
        return noisy_hu.astype(np.float32), noise.astype(np.float32)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        vol_idx, z = self.pairs[idx]
        vol = self.volumes[vol_idx]
        H, W, D = vol.shape

        # center, neighbor 선택
        if z == 0:
            z_pair = 1
        elif z == D - 1:
            z_pair = D - 2
        else:
            z_pair = z - 1 if random.random() < 0.5 else z + 1

        hu_center = vol[:, :, z]
        hu_pair = vol[:, :, z_pair]
        hu_mid = 0.5 * (hu_center + hu_pair)

        # 5-slice context (z-2 ~ z+2)
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

        # ★ NPS-guided synthetic noise: center slice에만 추가 ★
        if self.mode == "train":
            # Adaptive scale: slice-level NPS (noise_std)에 따라 증폭/감쇠
            scale = self.noise_aug_ratio
            if self.slice_noise_map and self.slice_noise_mean and self.volume_patient_ids:
                vol_idx, z = self.pairs[idx]
                pid = self.volume_patient_ids[vol_idx]
                key = (str(pid), int(z))
                noise_std = self.slice_noise_map.get(key, None)
                if noise_std is not None and self.slice_noise_mean > 0:
                    base = float(noise_std) / float(self.slice_noise_mean)
                    # base > 1: 이미 노이즈 많은 슬라이스 → 증폭 줄이고
                    # base < 1: 상대적으로 깨끗한 슬라이스 → 증폭 키우기
                    adaptive = 1.0 / (base + 1e-6)
                    adaptive = float(np.clip(adaptive, 0.7, 1.5))
                    scale = self.noise_aug_ratio * adaptive

            noisy_hu, noise_hu = self._add_ct_like_noise_nps_guided(
                hu_center, 
                scale=scale,
                augment_streaks=self.augment_streaks,
                streak_strength=self.streak_strength,
            )
            x_center_noisy = self._window_and_normalize(noisy_hu)
            noise_synthetic = (x_center_noisy - x_center).astype(np.float32)
        else:
            x_center_noisy = x_center.copy()
            noise_synthetic = np.zeros_like(x_center, dtype=np.float32)

        # 5-slice volume에서 center slice만 noisy 버전으로 교체
        x_5slices_aug = slices_5_norm.copy()
        x_5slices_aug[2] = x_center_noisy  # center index = 2

        # matched regions mask W
        body_mask = self._make_body_mask(hu_center)
        W = self._compute_match_map(x_center, x_pair, body_mask)

        # random crop
        x_5slices_aug, x_center, x_pair, x_mid, W, noise_synthetic = self._random_crop_5(
            x_5slices_aug, x_center, x_pair, x_mid, W, noise_synthetic
        )

        # flip augmentation (only for train mode)
        if self.mode == "train":
            x_5slices_aug, x_center, x_pair, x_mid, W, noise_synthetic = self._maybe_flip(
                x_5slices_aug, x_center, x_pair, x_mid, W, noise_synthetic
            )

        # contiguous arrays
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