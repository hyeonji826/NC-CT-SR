"""
dataset.py - MA-HybridNet 데이터 로딩
모드:
  - ActionGuided: Raw NC-CT + NPS noise → Streak-corrected target + masks
  - Noisier2Noise: NC-CT + NPS noise → NC-CT 원본 (fallback)
  - CECTTexturePatchPool: CE-CT DICOM에서 unpaired 텍스처 패치 풀
float32 정밀도 유지.
"""

import os
import random
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import SimpleITK as sitk
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

# ============================================================
# 상수 정의
# ============================================================
HU_MIN = -1000.0
HU_MAX = 1000.0
STREAK_CLIP_HU = 150.0   # streak map 정규화 클리핑 범위 (HU)
GLOBAL_SEED = 42


# ============================================================
# HU 정규화 / 역정규화
# ============================================================
def normalize_hu(hu_image: np.ndarray,
                 hu_min: float = HU_MIN,
                 hu_max: float = HU_MAX) -> np.ndarray:
    """HU → [-1, 1] 정규화. float32 정밀도 유지."""
    clipped = np.clip(hu_image, hu_min, hu_max).astype(np.float32)
    normalized = (clipped - hu_min) / (hu_max - hu_min)   # [0, 1]
    normalized = normalized * 2.0 - 1.0                    # [-1, 1]
    return normalized


def denormalize_hu(normalized: np.ndarray,
                   hu_min: float = HU_MIN,
                   hu_max: float = HU_MAX) -> np.ndarray:
    """[-1, 1] → HU 역변환."""
    rescaled = (normalized + 1.0) / 2.0
    hu_image = rescaled * (hu_max - hu_min) + hu_min
    return hu_image.astype(np.float32)


def normalize_streak(streak_hu: np.ndarray,
                     clip_hu: float = STREAK_CLIP_HU) -> np.ndarray:
    """Streak map HU → [-1, 1] 정규화. ±clip_hu로 클리핑."""
    return np.clip(streak_hu / clip_hu, -1.0, 1.0).astype(np.float32)


# ============================================================
# 유틸리티: 센터 크롭
# ============================================================
def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    """2D 영상을 중앙 기준으로 크롭."""
    H, W = img.shape
    ch = min(crop_size, H)
    cw = min(crop_size, W)
    y0 = (H - ch) // 2
    x0 = (W - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def load_nifti_volume_autoaxis(path: str) -> np.ndarray:
    """
    NIfTI를 로드하여 (Z, H, W) float32 HU 배열로 반환.
    Z축을 가장 작은 차원으로 자동 감지.
    """
    img = nib.load(str(path))
    arr = img.get_fdata().astype(np.float32)
    if arr.ndim == 3:
        z_axis = int(np.argmin(arr.shape))
        if z_axis == 2:
            arr = np.transpose(arr, (2, 0, 1))
        elif z_axis == 1:
            arr = np.transpose(arr, (1, 0, 2))
    return np.ascontiguousarray(arr)


# ============================================================
# 유틸: 환자 리스트 로드
# ============================================================
def _load_patient_list(split_dir: str, mode: str) -> List[str]:
    """data_splits/nc_ct_{mode}.txt에서 환자 ID 리스트 로드."""
    txt_path = Path(split_dir) / f"nc_ct_{mode}.txt"
    if not txt_path.exists():
        return []
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


# ============================================================
# Noisier2Noise: NC-CT + NPS 노이즈 → NC-CT 원본
# ============================================================
class NoisierFinetuneDataset(Dataset):
    """
    Noisier2Noise Finetune용 데이터셋:
    Input: NC-CT + NPS 합성 노이즈 (noisier), [-1, 1]
    Target: NC-CT 원본, [-1, 1]
    Streak map: pre-computed streak intensity (optional, 2nd input channel)

    학습 시: (NC-CT + noise) → NC-CT 학습
    추론 시: NC-CT → Model → 더 깨끗한 출력 기대
    """

    def __init__(self,
                 noisy_nifti_paths: List[str],
                 noise_model,
                 hu_min: float = HU_MIN,
                 hu_max: float = HU_MAX,
                 crop_size: int = 512,
                 split: str = "train",
                 seed: int = GLOBAL_SEED,
                 streak_root: str = ""):
        super().__init__()
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.crop_size = crop_size
        self.split = split
        self.noise_model = noise_model
        self.rng = random.Random(seed)
        self.streak_root = Path(streak_root) if streak_root else None

        # single-slot 캐시
        self._cache_path = None
        self._cache_vol = None
        self._cache_streak_pid = None
        self._cache_streak_vol = None

        # 슬라이스 인덱스 구성
        self.slice_infos = []
        for nii_path in noisy_nifti_paths:
            nii_path = Path(nii_path)
            if not nii_path.exists():
                print(f"[WARN] NIfTI not found: {nii_path}")
                continue

            vol = load_nifti_volume_autoaxis(str(nii_path))
            n_slices = vol.shape[0]
            patient_id = nii_path.stem
            start = int(n_slices * 0.05)
            end = int(n_slices * 0.95)
            for idx in range(start, max(start + 1, end)):
                self.slice_infos.append((str(nii_path), patient_id, idx))

        has_streak = self.streak_root and self.streak_root.is_dir()
        print(f"[Noisier2Noise-{split}] {len(noisy_nifti_paths)} vols, "
              f"{len(self.slice_infos)} slices, streak={'yes' if has_streak else 'no'}")

    def _get_slice(self, path: str, idx: int) -> np.ndarray:
        if self._cache_path != path:
            self._cache_vol = load_nifti_volume_autoaxis(path)
            self._cache_path = path
        return self._cache_vol[idx].copy()

    def _get_streak_slice(self, patient_id: str, idx: int) -> Optional[np.ndarray]:
        """Pre-computed streak map 로드. (Z, H, W) .npy"""
        if self.streak_root is None:
            return None
        if self._cache_streak_pid != patient_id:
            streak_path = self.streak_root / f"{patient_id}.npy"
            if streak_path.exists():
                self._cache_streak_vol = np.load(str(streak_path))
                self._cache_streak_pid = patient_id
            else:
                self._cache_streak_vol = None
                self._cache_streak_pid = patient_id
        if self._cache_streak_vol is None:
            return None
        if idx >= self._cache_streak_vol.shape[0]:
            return None
        return self._cache_streak_vol[idx].copy()

    def __len__(self) -> int:
        return len(self.slice_infos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        nii_path, patient_id, slice_idx = self.slice_infos[idx]

        # NC-CT 원본 (Target)
        target_hu = self._get_slice(nii_path, slice_idx)

        # Streak map (optional)
        streak_hu = self._get_streak_slice(patient_id, slice_idx)

        H, W = target_hu.shape

        # 크롭 처리
        if self.crop_size < min(H, W):
            if self.split == "train":
                y0 = self.rng.randint(0, H - self.crop_size)
                x0 = self.rng.randint(0, W - self.crop_size)
                target_hu = target_hu[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
                if streak_hu is not None:
                    streak_hu = streak_hu[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            else:
                target_hu = center_crop(target_hu, self.crop_size)
                if streak_hu is not None:
                    streak_hu = center_crop(streak_hu, self.crop_size)

        # NPS 노이즈 추가하여 Input 생성 (Noisier version)
        if self.noise_model is not None:
            input_hu = self.noise_model.add_noise(target_hu)
        else:
            input_hu = target_hu + np.random.randn(*target_hu.shape).astype(np.float32) * 50.0

        # 증강 적용 (train only)
        if self.split == "train":
            if self.rng.random() < 0.5:
                input_hu = np.flip(input_hu, axis=1).copy()
                target_hu = np.flip(target_hu, axis=1).copy()
                if streak_hu is not None:
                    streak_hu = np.flip(streak_hu, axis=1).copy()
            if self.rng.random() < 0.5:
                input_hu = np.flip(input_hu, axis=0).copy()
                target_hu = np.flip(target_hu, axis=0).copy()
                if streak_hu is not None:
                    streak_hu = np.flip(streak_hu, axis=0).copy()

        # HU 값 저장 (MA-HybridNet용)
        input_hu_copy = np.ascontiguousarray(input_hu.copy())
        target_hu_copy = np.ascontiguousarray(target_hu.copy())

        # [-1, 1] 정규화
        input_norm = np.ascontiguousarray(normalize_hu(input_hu, self.hu_min, self.hu_max))
        target_norm = np.ascontiguousarray(normalize_hu(target_hu, self.hu_min, self.hu_max))

        result = {
            'input': torch.from_numpy(input_norm).unsqueeze(0).float(),
            'target': torch.from_numpy(target_norm).unsqueeze(0).float(),
            'input_hu': torch.from_numpy(input_hu_copy).unsqueeze(0).float(),
            'target_hu': torch.from_numpy(target_hu_copy).unsqueeze(0).float(),
        }

        # Streak map → [-1, 1] 정규화 (±150 HU → ±1.0)
        if streak_hu is not None:
            streak_norm = np.ascontiguousarray(normalize_streak(streak_hu))
            result['streak_map'] = torch.from_numpy(streak_norm).unsqueeze(0).float()

        return result


def get_noisier_finetune_dataloaders(noisy_root: str,
                                      noise_model,
                                      split_dir: str,
                                      crop_size: int = 512,
                                      batch_size: int = 2,
                                      num_workers: int = 0,
                                      seed: int = GLOBAL_SEED,
                                      streak_root: str = "",
                                      ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Noisier2Noise Finetune Train/Val/Test 데이터로더 생성.
    Input: NC-CT + NPS noise (noisier)
    Target: NC-CT original
    """
    noisy_root_p = Path(noisy_root)

    train_ids = _load_patient_list(split_dir, "train")
    val_ids = _load_patient_list(split_dir, "val")
    test_ids = _load_patient_list(split_dir, "test")

    def resolve_paths(ids):
        paths = []
        for pid in ids:
            p = noisy_root_p / f"{pid}.nii"
            if p.exists():
                paths.append(str(p))
            else:
                p_gz = noisy_root_p / f"{pid}.nii.gz"
                if p_gz.exists():
                    paths.append(str(p_gz))
        return paths

    train_files = resolve_paths(train_ids)
    val_files = resolve_paths(val_ids)
    test_files = resolve_paths(test_ids)

    print(f"[Noisier2Noise] Patients - Train: {len(train_files)}, "
          f"Val: {len(val_files)}, Test: {len(test_files)}")

    train_ds = NoisierFinetuneDataset(
        train_files, noise_model,
        crop_size=crop_size, split="train", seed=seed,
        streak_root=streak_root,
    )
    val_ds = NoisierFinetuneDataset(
        val_files, noise_model,
        crop_size=crop_size, split="val", seed=seed,
        streak_root=streak_root,
    )
    test_ds = NoisierFinetuneDataset(
        test_files, noise_model,
        crop_size=crop_size, split="test", seed=seed,
        streak_root=streak_root,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader


# ============================================================
# Action-Guided Dataset: Raw NC-CT input → Streak-corrected target
# ============================================================

def _mild_target_smooth(image_hu: np.ndarray) -> np.ndarray:
    """Target에 3x3 median filter 적용 (양자 노이즈 경감, 구조 보존)."""
    from scipy.ndimage import median_filter
    return median_filter(image_hu, size=3).astype(np.float32)


class ActionGuidedDataset(Dataset):
    """
    Action-Guided Training Dataset.

    Input:  raw NC-CT + NPS synthetic noise  (noisier)
    Target: streak-corrected NC-CT           (pseudo-clean)

    핵심:
    - Target에서 arm streak artifact가 제거됨
    - 모델은 N2N으로 노이즈 제거 + streak 보정을 동시에 학습
    - Input은 원본 NC-CT 기반 (inference와 동일 분포)
    """

    def __init__(self,
                 raw_nifti_paths: List[str],
                 processed_nifti_paths: List[str],
                 noise_model,
                 mask_root: str = "",
                 hu_min: float = HU_MIN,
                 hu_max: float = HU_MAX,
                 crop_size: int = 512,
                 split: str = "train",
                 seed: int = GLOBAL_SEED,
                 streak_root: str = ""):
        super().__init__()
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.crop_size = crop_size
        self.split = split
        self.noise_model = noise_model
        self.mask_root = Path(mask_root) if mask_root else None
        self.streak_root = Path(streak_root) if streak_root else None
        self.rng = random.Random(seed)

        # Single-slot cache (per source)
        self._cache_raw_path = None
        self._cache_raw_vol = None
        self._cache_proc_path = None
        self._cache_proc_vol = None
        self._cache_mask_pid = None
        self._cache_masks = None
        self._cache_streak_pid = None
        self._cache_streak_vol = None

        # Build (raw_path, processed_path, patient_id, slice_idx) tuples
        self.slice_infos = []

        for raw_path, proc_path in zip(raw_nifti_paths, processed_nifti_paths):
            raw_p = Path(raw_path)
            proc_p = Path(proc_path)

            if not raw_p.exists():
                print(f"[WARN] Raw NIfTI not found: {raw_p}")
                continue
            if not proc_p.exists():
                print(f"[WARN] Processed NIfTI not found: {proc_p}")
                continue

            patient_id = raw_p.stem  # e.g. "1728852"
            vol = load_nifti_volume_autoaxis(str(raw_p))
            n_slices = vol.shape[0]
            start = int(n_slices * 0.05)
            end = int(n_slices * 0.95)
            for idx in range(start, max(start + 1, end)):
                self.slice_infos.append((str(raw_p), str(proc_p), patient_id, idx))

        has_masks = self.mask_root and self.mask_root.is_dir()
        print(f"[ActionGuided-{split}] {len(raw_nifti_paths)} vols, "
              f"{len(self.slice_infos)} slices, masks={'yes' if has_masks else 'no'}")

    def _get_raw_slice(self, path: str, idx: int) -> np.ndarray:
        if self._cache_raw_path != path:
            self._cache_raw_vol = load_nifti_volume_autoaxis(path)
            self._cache_raw_path = path
        return self._cache_raw_vol[idx].copy()

    def _get_proc_slice(self, path: str, idx: int) -> np.ndarray:
        if self._cache_proc_path != path:
            self._cache_proc_vol = load_nifti_volume_autoaxis(path)
            self._cache_proc_path = path
        return self._cache_proc_vol[idx].copy()

    def _get_mask_slices(self, patient_id: str, slice_idx: int):
        """
        Pre-computed action mask 로드 (.npz).
        Returns: dict with 'artifact', 'fluid', 'structure' as float32 [0, 1]
                 or None if masks not available.
        """
        if self.mask_root is None:
            return None

        # Cache: 같은 patient면 재로드 생략
        if self._cache_mask_pid != patient_id:
            mask_path = self.mask_root / f"{patient_id}.npz"
            if mask_path.exists():
                data = np.load(str(mask_path))
                self._cache_masks = {
                    'artifact': data['artifact'],    # (Z, H, W) uint8
                    'fluid': data['fluid'],
                    'structure': data['structure'],
                }
                self._cache_mask_pid = patient_id
            else:
                self._cache_masks = None
                self._cache_mask_pid = patient_id

        if self._cache_masks is None:
            return None

        return {
            k: v[slice_idx].astype(np.float32) / 255.0
            for k, v in self._cache_masks.items()
        }

    def _get_streak_slice(self, patient_id: str, slice_idx: int) -> Optional[np.ndarray]:
        """Pre-computed streak map 로드. (Z, H, W) .npy"""
        if self.streak_root is None:
            return None
        if self._cache_streak_pid != patient_id:
            streak_path = self.streak_root / f"{patient_id}.npy"
            if streak_path.exists():
                self._cache_streak_vol = np.load(str(streak_path))
                self._cache_streak_pid = patient_id
            else:
                self._cache_streak_vol = None
                self._cache_streak_pid = patient_id
        if self._cache_streak_vol is None:
            return None
        if slice_idx >= self._cache_streak_vol.shape[0]:
            return None
        return self._cache_streak_vol[slice_idx].copy()

    def __len__(self) -> int:
        return len(self.slice_infos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raw_path, proc_path, patient_id, slice_idx = self.slice_infos[idx]

        # Raw NC-CT (input basis)
        raw_hu = self._get_raw_slice(raw_path, slice_idx)

        # Processed NC-CT (target: streak-corrected + mild denoise)
        proc_hu = self._get_proc_slice(proc_path, slice_idx)

        # Streak map (optional)
        streak_hu = self._get_streak_slice(patient_id, slice_idx)

        # Inpaint mask: |raw - processed| → 실제 수정된 픽셀 (smooth 전에 계산)
        inpaint_diff = np.abs(raw_hu - proc_hu)

        target_hu = _mild_target_smooth(proc_hu)

        # Action masks (pre-computed)
        masks = self._get_mask_slices(patient_id, slice_idx)

        H, W = raw_hu.shape

        # Crop (same coords for input, target, masks, inpaint_diff)
        if self.crop_size < min(H, W):
            if self.split == "train":
                y0 = self.rng.randint(0, H - self.crop_size)
                x0 = self.rng.randint(0, W - self.crop_size)
            else:
                y0 = (H - self.crop_size) // 2
                x0 = (W - self.crop_size) // 2
            raw_hu = raw_hu[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            target_hu = target_hu[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            inpaint_diff = inpaint_diff[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            if streak_hu is not None:
                streak_hu = streak_hu[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
            if masks is not None:
                masks = {k: v[y0:y0 + self.crop_size, x0:x0 + self.crop_size]
                         for k, v in masks.items()}

        # Input: raw + synthetic NPS noise
        if self.noise_model is not None:
            input_hu = self.noise_model.add_noise(raw_hu)
        else:
            input_hu = raw_hu + np.random.randn(*raw_hu.shape).astype(np.float32) * 50.0

        # Augmentation (train only, same for all spatial arrays)
        if self.split == "train":
            if self.rng.random() < 0.5:
                input_hu = np.flip(input_hu, axis=1).copy()
                target_hu = np.flip(target_hu, axis=1).copy()
                inpaint_diff = np.flip(inpaint_diff, axis=1).copy()
                if streak_hu is not None:
                    streak_hu = np.flip(streak_hu, axis=1).copy()
                if masks is not None:
                    masks = {k: np.flip(v, axis=1).copy() for k, v in masks.items()}
            if self.rng.random() < 0.5:
                input_hu = np.flip(input_hu, axis=0).copy()
                target_hu = np.flip(target_hu, axis=0).copy()
                inpaint_diff = np.flip(inpaint_diff, axis=0).copy()
                if streak_hu is not None:
                    streak_hu = np.flip(streak_hu, axis=0).copy()
                if masks is not None:
                    masks = {k: np.flip(v, axis=0).copy() for k, v in masks.items()}

        # HU copies (for model's HU classifier)
        input_hu_copy = input_hu.copy()
        target_hu_copy = target_hu.copy()

        # Normalize to [-1, 1]
        input_norm = normalize_hu(input_hu, self.hu_min, self.hu_max)
        target_norm = normalize_hu(target_hu, self.hu_min, self.hu_max)

        input_norm = np.ascontiguousarray(input_norm)
        target_norm = np.ascontiguousarray(target_norm)
        input_hu_copy = np.ascontiguousarray(input_hu_copy)
        target_hu_copy = np.ascontiguousarray(target_hu_copy)

        # Inpaint mask: HU diff → [0, 1] (soft mask, 연속값)
        # threshold 5 HU 이하 = 수정 안 됨 (0), 50 HU 이상 = 확실히 수정됨 (1)
        inpaint_mask = np.clip((inpaint_diff - 5.0) / 45.0, 0, 1).astype(np.float32)
        inpaint_mask = np.ascontiguousarray(inpaint_mask)

        result = {
            'input': torch.from_numpy(input_norm).unsqueeze(0).float(),
            'target': torch.from_numpy(target_norm).unsqueeze(0).float(),
            'input_hu': torch.from_numpy(input_hu_copy).unsqueeze(0).float(),
            'target_hu': torch.from_numpy(target_hu_copy).unsqueeze(0).float(),
            'inpaint_mask': torch.from_numpy(inpaint_mask).unsqueeze(0).float(),
        }

        # Streak map → (1, H, W) normalized [-1, 1]
        if streak_hu is not None:
            streak_norm = np.ascontiguousarray(normalize_streak(streak_hu))
            result['streak_map'] = torch.from_numpy(streak_norm).unsqueeze(0).float()

        # Action masks → (1, H, W) float tensors
        if masks is not None:
            for k in ('artifact', 'fluid', 'structure'):
                m = np.ascontiguousarray(masks[k])
                result[f'mask_{k}'] = torch.from_numpy(m).unsqueeze(0).float()

        return result


def get_action_guided_dataloaders(
    raw_root: str,
    processed_root: str,
    noise_model,
    split_dir: str,
    mask_root: str = "",
    crop_size: int = 512,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = GLOBAL_SEED,
    streak_root: str = "",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Action-Guided Training 데이터로더 생성.

    Input: raw NC-CT + NPS noise
    Target: streak-corrected NC-CT (pre-computed)
    Masks: artifact/fluid/structure (pre-computed, optional)
    """
    raw_root_p = Path(raw_root)
    proc_root_p = Path(processed_root)

    train_ids = _load_patient_list(split_dir, "train")
    val_ids = _load_patient_list(split_dir, "val")
    test_ids = _load_patient_list(split_dir, "test")

    def resolve_paired_paths(ids):
        raw_paths, proc_paths = [], []
        for pid in ids:
            raw_p = raw_root_p / f"{pid}.nii"
            if not raw_p.exists():
                raw_p = raw_root_p / f"{pid}.nii.gz"

            proc_p = proc_root_p / f"{pid}.nii"
            if not proc_p.exists():
                proc_p = proc_root_p / f"{pid}.nii.gz"

            if raw_p.exists() and proc_p.exists():
                raw_paths.append(str(raw_p))
                proc_paths.append(str(proc_p))
            elif raw_p.exists():
                # Processed not available → fallback to raw as target
                raw_paths.append(str(raw_p))
                proc_paths.append(str(raw_p))
                print(f"[WARN] No processed file for {pid}, using raw as target")
        return raw_paths, proc_paths

    train_raw, train_proc = resolve_paired_paths(train_ids)
    val_raw, val_proc = resolve_paired_paths(val_ids)
    test_raw, test_proc = resolve_paired_paths(test_ids)

    print(f"[ActionGuided] Patients - Train: {len(train_raw)}, "
          f"Val: {len(val_raw)}, Test: {len(test_raw)}")

    train_ds = ActionGuidedDataset(
        train_raw, train_proc, noise_model, mask_root=mask_root,
        crop_size=crop_size, split="train", seed=seed,
        streak_root=streak_root,
    )
    val_ds = ActionGuidedDataset(
        val_raw, val_proc, noise_model, mask_root=mask_root,
        crop_size=crop_size, split="val", seed=seed,
        streak_root=streak_root,
    )
    test_ds = ActionGuidedDataset(
        test_raw, test_proc, noise_model, mask_root=mask_root,
        crop_size=crop_size, split="test", seed=seed,
        streak_root=streak_root,
    )

    ag_train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    ag_val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    ag_test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return ag_train_loader, ag_val_loader, ag_test_loader


# ============================================================
# CE-CT Texture Patch Pool (Unpaired, Gradient-Domain Discriminator용)
# ============================================================

class CECTTexturePatchPool:
    """
    CE-CT DICOM에서 랜덤 슬라이스를 추출하여 in-memory 텍스처 패치 풀 구축.
    Discriminator에 "real CT texture" 분포를 제공.

    완전 unpaired: patient/slice 매칭 불필요.
    Gradient domain에서 작동하므로 조영 효과 차이는 자동 무시.
    """

    def __init__(self,
                 dicom_root: str,
                 pool_size: int = 500,
                 crop_size: int = 512,
                 hu_min: float = HU_MIN,
                 hu_max: float = HU_MAX,
                 seed: int = GLOBAL_SEED):
        if not HAS_SITK:
            raise ImportError("SimpleITK is required for CECTTexturePatchPool")

        self.crop_size = crop_size
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.rng = random.Random(seed)

        dicom_root_p = Path(dicom_root)
        patient_dirs = sorted([d for d in dicom_root_p.iterdir() if d.is_dir()])

        if not patient_dirs:
            print(f"[CECTPool] WARNING: No patient dirs in {dicom_root}")
            self.pool = None
            return

        self.rng.shuffle(patient_dirs)

        print(f"[CECTPool] Building texture pool from {len(patient_dirs)} patients...")
        patches = []
        attempts = 0
        max_attempts = pool_size * 3

        while len(patches) < pool_size and attempts < max_attempts:
            attempts += 1
            patient_dir = patient_dirs[attempts % len(patient_dirs)]

            try:
                vol = self._load_random_volume(patient_dir)
                if vol is None or vol.shape[0] < 5:
                    continue

                # 랜덤 슬라이스 (상하 10% 제외)
                z_start = int(vol.shape[0] * 0.1)
                z_end = int(vol.shape[0] * 0.9)
                if z_end <= z_start:
                    continue
                z = self.rng.randint(z_start, z_end - 1)
                hu_slice = vol[z]

                # 센터 크롭
                patch = center_crop(hu_slice, crop_size)
                if patch.shape[0] < crop_size or patch.shape[1] < crop_size:
                    continue

                # body 영역 체크 (너무 빈 슬라이스 제외)
                body_ratio = np.mean(patch > -500)
                if body_ratio < 0.3:
                    continue

                # [-1, 1] 정규화
                patch_norm = normalize_hu(patch, hu_min, hu_max)
                patches.append(patch_norm)

            except Exception:
                continue

        if not patches:
            print(f"[CECTPool] WARNING: Failed to build pool")
            self.pool = None
            return

        # (N, 1, H, W) float32 tensor
        self.pool = torch.from_numpy(
            np.stack(patches)[:, np.newaxis, :, :]
        ).float()

        print(f"[CECTPool] Pool ready: {self.pool.shape[0]} patches, "
              f"{self.pool.element_size() * self.pool.nelement() / 1024 / 1024:.0f} MB")

    def _load_random_volume(self, patient_dir: Path) -> Optional[np.ndarray]:
        """DICOM 시리즈를 HU 볼륨 (Z, H, W) float32로 로드."""
        series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(patient_dir))
        if not series_IDs:
            return None

        # 가장 긴 시리즈 선택
        best_sid, best_n = None, 0
        for sid in series_IDs:
            files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(patient_dir), sid)
            if len(files) > best_n:
                best_n = len(files)
                best_sid = sid

        if best_sid is None or best_n < 5:
            return None

        dcm_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(patient_dir), best_sid)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dcm_files)
        img = reader.Execute()
        return sitk.GetArrayFromImage(img).astype(np.float32)

    def sample(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        랜덤 패치 batch_size개 샘플링 (augmentation 포함).

        Returns:
            (B, 1, H, W) tensor on device, or None if pool unavailable
        """
        if self.pool is None:
            return None

        indices = [self.rng.randint(0, self.pool.shape[0] - 1)
                   for _ in range(batch_size)]
        batch = self.pool[indices].clone()

        # 랜덤 augmentation (flip/rot90) - 텍스처 다양성 확보
        for i in range(batch_size):
            if self.rng.random() < 0.5:
                batch[i] = batch[i].flip(-1)
            if self.rng.random() < 0.5:
                batch[i] = batch[i].flip(-2)
            rot_k = self.rng.randint(0, 3)
            if rot_k > 0:
                batch[i] = torch.rot90(batch[i], rot_k, [-2, -1])

        return batch.to(device)