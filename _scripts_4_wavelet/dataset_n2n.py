# dataset_n2n.py - Self-Supervised Dataset for Neighbor2Neighbor

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset
import random
from scipy.ndimage import rotate


class NCCTDenoiseDataset(Dataset):
    """
    NC-CT Dataset for Self-Supervised Learning (Neighbor2Neighbor)
    
    Key Difference from Supervised:
    - NO paired data required!
    - Only noisy images (NC-CT)
    - N2N creates its own "target" via subsampling
    
    Data Loading:
    - On-the-fly loading (memory efficient)
    - Random patch extraction
    - Data augmentation
    - Background filtering
    """
    
    def __init__(self, 
                 nc_ct_dir,
                 hu_window=(-160, 240),
                 patch_size=128,
                 config_aug=None,
                 mode='train'):
        
        self.nc_ct_dir = Path(nc_ct_dir)
        self.hu_window = hu_window
        self.patch_size = patch_size
        self.mode = mode
        
        # Augmentation config
        self.aug_config = config_aug if config_aug and mode == 'train' else {}
        
        # Get all NC-CT files
        self.files = sorted(list(self.nc_ct_dir.glob("*.nii.gz")))
        
        if len(self.files) == 0:
            raise ValueError(f"No NIfTI files found in {self.nc_ct_dir}")
        
        print(f"\n📁 [{mode}] NC-CT Dataset:")
        print(f"   Files found: {len(self.files)}")
        print(f"   HU window: {self.hu_window}")
        print(f"   Patch size: {patch_size}")
        print(f"   Mode: {mode}")
        print(f"   → Self-supervised (no pairs needed!)")
    
    def normalize_hu(self, img):
        """
        HU windowing and normalization to [0, 1]
        
        Args:
            img: numpy array with HU values
            
        Returns:
            normalized: [0, 1] float32 array
        """
        # Clip to window
        img = np.clip(img, self.hu_window[0], self.hu_window[1])
        
        # Normalize to [0, 1]
        img = (img - self.hu_window[0]) / (self.hu_window[1] - self.hu_window[0])
        
        # Safety checks
        if np.isnan(img).any():
            print("⚠️ Warning: NaN detected, filling with 0")
            img = np.nan_to_num(img, 0)
        if np.isinf(img).any():
            print("⚠️ Warning: Inf detected, filling with 0")
            img = np.nan_to_num(img, 0)
        
        return img.astype(np.float32)
    
    def is_valid_patch(self, patch, min_std=0.01, min_mean=0.05):
        """
        Check if patch contains meaningful content (not just background/air)
        
        Args:
            patch: normalized image patch [0, 1]
            min_std: minimum standard deviation
            min_mean: minimum mean intensity
            
        Returns:
            valid: True if patch has content
        """
        # Too uniform (background)
        if patch.std() < min_std:
            return False
        
        # Too dark (air)
        if patch.mean() < min_mean:
            return False
        
        return True
    
    def random_crop(self, volume, max_attempts=10):
        """
        Extract random valid patch from 3D volume
        
        Args:
            volume: 3D numpy array [H, W, D]
            max_attempts: maximum attempts to find valid patch
            
        Returns:
            patch: 2D patch [patch_size, patch_size]
        """
        h, w, d = volume.shape
        
        for attempt in range(max_attempts):
            # Random slice
            slice_idx = random.randint(0, d - 1)
            slice_2d = volume[:, :, slice_idx]
            
            # Random crop if image is larger than patch_size
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                patch = slice_2d[top:top+self.patch_size, left:left+self.patch_size]
            else:
                # Center crop or pad if needed
                patch = slice_2d
            
            # Normalize
            patch = self.normalize_hu(patch)
            
            # Check validity
            if self.is_valid_patch(patch):
                return patch
        
        # If all attempts failed, return last patch anyway
        return patch
    
    def augment(self, patch):
        """
        Data augmentation
        
        Args:
            patch: 2D image patch
            
        Returns:
            augmented: augmented patch
        """
        # Horizontal flip
        if random.random() < self.aug_config.get('horizontal_flip', 0.5):
            patch = np.fliplr(patch).copy()
        
        # Vertical flip
        if random.random() < self.aug_config.get('vertical_flip', 0.3):
            patch = np.flipud(patch).copy()
        
        # Rotation
        rotation_range = self.aug_config.get('rotation_range', 0)
        if rotation_range > 0 and random.random() < 0.5:
            angle = random.uniform(-rotation_range, rotation_range)
            patch = rotate(patch, angle, reshape=False, order=1)
            patch = np.clip(patch, 0, 1)
        
        # Small noise augmentation (increases diversity)
        noise_std = self.aug_config.get('noise_std', 0)
        if noise_std > 0 and random.random() < 0.3:
            noise = np.random.normal(0, noise_std, patch.shape).astype(np.float32)
            patch = np.clip(patch + noise, 0, 1)
        
        return patch
    
    def __len__(self):
        """
        Dataset length
        
        For training: multiple patches per volume
        For validation: fewer patches
        """
        if self.mode == 'train':
            return len(self.files) * 100  # 100 patches per volume
        else:
            return len(self.files) * 10  # 10 patches per volume
    
    def __getitem__(self, idx):
        """
        Get a single sample (noisy patch)
        
        Returns:
            noisy_patch: [1, H, W] tensor
            
        Note: N2N doesn't need target! It creates target via subsampling
        """
        # Select volume (cycle through files)
        vol_idx = idx % len(self.files)
        file_path = self.files[vol_idx]
        
        # Load volume on-the-fly (memory efficient)
        try:
            nii = nib.load(str(file_path))
            volume = nii.get_fdata()
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            # Return a dummy tensor if loading fails
            return torch.zeros(1, self.patch_size, self.patch_size)
        
        # Extract random patch
        patch = self.random_crop(volume)
        
        # Augmentation (train only)
        if self.mode == 'train' and self.aug_config:
            patch = self.augment(patch)
        
        # Convert to tensor [1, H, W]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        
        return patch_tensor


class DualNCCTDataset(NCCTDenoiseDataset):
    """
    Dual NC-CT Dataset (for validation with reference if available)
    
    If you have some paired data for validation metrics only:
    - Still trains self-supervised (N2N)
    - But can compute PSNR/SSIM against reference
    
    Optional: can be used same as NCCTDenoiseDataset
    """
    
    def __init__(self, nc_ct_dir, reference_dir=None, **kwargs):
        super().__init__(nc_ct_dir, **kwargs)
        
        self.reference_dir = Path(reference_dir) if reference_dir else None
        
        if self.reference_dir and self.reference_dir.exists():
            self.ref_files = sorted(list(self.reference_dir.glob("*.nii.gz")))
            print(f"   Reference dir: {reference_dir}")
            print(f"   Reference files: {len(self.ref_files)}")
        else:
            self.ref_files = None
            print(f"   No reference dir (self-supervised only)")
    
    def __getitem__(self, idx):
        """
        Returns:
            noisy_patch: [1, H, W]
            ref_patch: [1, H, W] if available, else same as noisy
        """
        noisy_patch = super().__getitem__(idx)
        
        # If no reference, return noisy as both
        if self.ref_files is None or len(self.ref_files) == 0:
            return noisy_patch, noisy_patch
        
        # Load reference (same patient, same slice)
        vol_idx = idx % len(self.files)
        
        try:
            ref_file = self.ref_files[vol_idx]
            ref_nii = nib.load(str(ref_file))
            ref_vol = ref_nii.get_fdata()
            ref_patch = self.random_crop(ref_vol)
            ref_tensor = torch.from_numpy(ref_patch).unsqueeze(0)
            return noisy_patch, ref_tensor
        except:
            # If reference loading fails, use noisy as reference
            return noisy_patch, noisy_patch