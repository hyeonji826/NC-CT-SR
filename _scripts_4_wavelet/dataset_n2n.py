import random
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class NCCTDenoiseDataset(Dataset):
    def __init__(
        self,
        nc_ct_dir,
        hu_window=(-160, 240),
        patch_size=128,
        mode="train",
        min_body_fraction=0.05,
    ):
        super().__init__()
        self.nc_ct_dir = Path(nc_ct_dir)
        self.hu_min, self.hu_max = hu_window
        self.patch_size = patch_size
        self.mode = mode
        self.min_body_fraction = min_body_fraction

        self.files = sorted(
            list(self.nc_ct_dir.glob("*.nii.gz")) +
            list(self.nc_ct_dir.glob("*.nii"))
        )
        if len(self.files) == 0:
            raise FileNotFoundError(f"No NIfTI files in {self.nc_ct_dir}")

        self.volumes = []
        self.slice_index = []

        print(f"\n📂 Loading NC-CT volumes...")
        for vol_idx, path in enumerate(self.files):
            nii = nib.load(str(path))
            vol = nii.get_fdata().astype(np.float32)
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

            H, W, D = vol.shape
            self.volumes.append(vol)

            for z in range(1, D - 1):
                slice_2d = vol[:, :, z]
                body_mask = (slice_2d > -500) & (slice_2d < 500)
                body_frac = body_mask.sum() / float(H * W)
                if body_frac < self.min_body_fraction:
                    continue
                self.slice_index.append((vol_idx, z))

        if len(self.slice_index) == 0:
            raise RuntimeError("No valid slices found.")

        print(f"   {len(self.files)} volumes, {len(self.slice_index)} slices (mode={self.mode})")

    def _window_and_normalize(self, slice_2d: np.ndarray) -> np.ndarray:
        s = np.clip(slice_2d, self.hu_min, self.hu_max)
        s = (s - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
        return s.astype(np.float32)

    def _random_crop(self, arr: np.ndarray) -> np.ndarray:
        if self.patch_size is None:
            return arr
        H, W = arr.shape[-2:]
        if H <= self.patch_size or W <= self.patch_size:
            return arr

        top = random.randint(0, H - self.patch_size)
        left = random.randint(0, W - self.patch_size)
        return arr[..., top:top + self.patch_size, left:left + self.patch_size]

    def _calculate_weight_matrix(self, s_curr: np.ndarray, s_next: np.ndarray) -> np.ndarray:
        """
        NS-N2N weight matrix with adaptive threshold
        
        저선량 CT 특성:
        - Noise: ~40 HU
        - Slice diff: ~20 HU
        - Normalized (0-1): threshold ≈ 0.05 (20/400)
        """
        from scipy.ndimage import median_filter
        
        # Low-pass filtering (median 3x3)
        lpf_curr = median_filter(s_curr, size=3)
        lpf_next = median_filter(s_next, size=3)
        
        # Residual
        residual = np.abs(lpf_curr - lpf_next)
        
        # Adaptive threshold (normalized 기준)
        # HU window 400 기준, 20 HU ≈ 0.05
        threshold = 0.05
        
        # Weight matrix
        weight = (residual <= threshold).astype(np.float32)
        
        return weight

    def __getitem__(self, idx):
        vol_idx, z = self.slice_index[idx]
        vol = self.volumes[vol_idx]

        s_prev = self._window_and_normalize(vol[:, :, z - 1])
        s_curr = self._window_and_normalize(vol[:, :, z])
        s_next = self._window_and_normalize(vol[:, :, z + 1])

        # NS-N2N: curr → next 예측, matched region에서만
        # Input: 3D context
        stack = np.stack([s_prev, s_curr, s_next], axis=0)
        
        # Target: next slice
        target = s_next
        
        # Weight: curr vs next matched
        weight = self._calculate_weight_matrix(s_curr, s_next)
        
        # Crop
        stack = self._random_crop(stack)
        target = self._random_crop(target)
        weight = self._random_crop(weight)

        input_tensor = torch.from_numpy(stack)  # [3, H, W]
        target_tensor = torch.from_numpy(target).unsqueeze(0)  # [1, H, W]
        weight_tensor = torch.from_numpy(weight).unsqueeze(0)  # [1, H, W]

        return input_tensor, target_tensor, weight_tensor

    def __len__(self):
        return len(self.slice_index)