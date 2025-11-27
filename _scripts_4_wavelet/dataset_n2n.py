# E:\LD-CT SR\_scripts_4_wavelet\dataset_n2n.py
# 2.5D Neighboring-Slice Self-Supervised Dataset for SwinIR
#
# 입력 : [3, H, W]  (z-1, z, z+1 슬라이스 스택)
# 타깃 : [1, H, W]  (중앙 슬라이스 z)
#
# - HU 윈도우링 후 [0,1] 정규화
# - 랜덤 패치 크롭 (patch_size)
# - train/val/test 모두 같은 Dataset 클래스 사용 (random_split)

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
        """
        nc_ct_dir: NIfTI(.nii/.nii.gz) NC-CT 볼륨 폴더
        hu_window: (minHU, maxHU)
        patch_size: 정사각형 패치 크기
        mode: 'train' / 'val' / 'test' (현재는 동작 동일, 추후 필요 시 분기 가능)
        min_body_fraction: slice 안에서 body(조직) 비율이 이 값보다 작으면 제외
        """
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
            raise FileNotFoundError(f"No NIfTI files found in {self.nc_ct_dir}")

        # 메모리에 전부 적재 (데이터 수가 크지 않다는 전제; 지금 프로젝트 규모면 OK)
        self.volumes = []
        self.slice_index = []  # (vol_idx, z)

        print(f"\n📂 Loading NC-CT volumes for NS-N2N dataset...")
        for vol_idx, path in enumerate(self.files):
            nii = nib.load(str(path))
            vol = nii.get_fdata().astype(np.float32)
            # NaN/inf 방지
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

            H, W, D = vol.shape
            self.volumes.append(vol)

            # z=1 ~ D-2 만 사용 (양 끝은 이웃 슬라이스 부족)
            for z in range(1, D - 1):
                slice_2d = vol[:, :, z]

                # body mask: 약한 조건으로 대략적인 신체 영역만 남김
                body_mask = (slice_2d > -500) & (slice_2d < 500)
                body_frac = body_mask.sum() / float(H * W)
                if body_frac < self.min_body_fraction:
                    continue

                self.slice_index.append((vol_idx, z))

        if len(self.slice_index) == 0:
            raise RuntimeError("No valid slices found for NS-N2N dataset.")

        print(f"   Loaded {len(self.files)} volumes, "
              f"{len(self.slice_index)} usable slices (mode={self.mode})")

    def __len__(self):
        return len(self.slice_index)

    def _window_and_normalize(self, slice_2d: np.ndarray) -> np.ndarray:
        s = np.clip(slice_2d, self.hu_min, self.hu_max)
        s = (s - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
        return s.astype(np.float32)

    def _random_crop(self, arr: np.ndarray) -> np.ndarray:
        """arr: [..., H, W]"""
        if self.patch_size is None:
            return arr
        H, W = arr.shape[-2:]
        if H <= self.patch_size or W <= self.patch_size:
            return arr

        top = random.randint(0, H - self.patch_size)
        left = random.randint(0, W - self.patch_size)
        return arr[..., top:top + self.patch_size, left:left + self.patch_size]

    def __getitem__(self, idx):
        vol_idx, z = self.slice_index[idx]
        vol = self.volumes[vol_idx]

        s0 = self._window_and_normalize(vol[:, :, z - 1])
        s1 = self._window_and_normalize(vol[:, :, z])
        s2 = self._window_and_normalize(vol[:, :, z + 1])

        # [3, H, W], [H, W]
        stack = np.stack([s0, s1, s2], axis=0)
        target = s1  # 중앙 슬라이스

        # 랜덤 패치 크롭 (train/val/test 모두 동일하게; random_split으로 나뉘므로 OK)
        stack = self._random_crop(stack)
        target = self._random_crop(target)

        input_tensor = torch.from_numpy(stack)              # [3, H, W]
        target_tensor = torch.from_numpy(target).unsqueeze(0)  # [1, H, W]

        return input_tensor, target_tensor
