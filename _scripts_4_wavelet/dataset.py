import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset
import random
from scipy.ndimage import rotate

class CTDenoiseDataset(Dataset):
    def __init__(self, low_dose_dir, full_dose_dir, 
                 hu_window=(-160, 240), patch_size=128, 
                 config_aug=None, mode='train'):
        
        self.low_dose_dir = Path(low_dose_dir)
        self.full_dose_dir = Path(full_dose_dir)
        self.hu_window = hu_window
        self.patch_size = patch_size
        self.mode = mode
        
        # Augmentation config
        self.aug_config = config_aug if config_aug and mode == 'train' else {}
        
        # Get paired files
        low_files = sorted(list(self.low_dose_dir.glob("*.nii.gz")))
        full_files = sorted(list(self.full_dose_dir.glob("*.nii.gz")))
        
        # Match by patient ID
        self.pairs = []
        for lf in low_files:
            patient_id = lf.stem.split('_')[0]
            matching_full = [f for f in full_files if patient_id in f.stem]
            if matching_full:
                self.pairs.append((lf, matching_full[0]))
        
        print(f"[{mode}] Found {len(self.pairs)} paired volumes")
        
        # ========== 개선: On-the-fly 로딩 (메모리 절약!) ==========
        # Pre-load 하지 않고 파일 경로만 저장
        # self.volumes = [] 제거
        # ========================================================
    
    def normalize_hu(self, img):
        """HU clipping and normalization to [0, 1]"""
        img = np.clip(img, self.hu_window[0], self.hu_window[1])
        img = (img - self.hu_window[0]) / (self.hu_window[1] - self.hu_window[0])
        
        # Check for nan/inf
        if np.isnan(img).any():
            print("⚠️ Warning: NaN in normalized image!")
            img = np.nan_to_num(img, 0)
        if np.isinf(img).any():
            print("⚠️ Warning: Inf in normalized image!")
            img = np.nan_to_num(img, 0)
        
        return img.astype(np.float32)
    
    def is_valid_patch(self, patch, min_std=0.01, min_mean=0.05):
        """Check if patch has meaningful content (not all background)"""
        # Skip patches that are mostly background/air
        if patch.std() < min_std:  # Too uniform
            return False
        if patch.mean() < min_mean:  # Too dark (mostly air)
            return False
        return True
    
    def random_crop(self, low_vol, full_vol, max_attempts=10):
        """Extract random patch with content validation"""
        h, w, d = low_vol.shape
        
        for attempt in range(max_attempts):
            # Random slice
            slice_idx = random.randint(0, d - 1)
            
            low_slice = low_vol[:, :, slice_idx]
            full_slice = full_vol[:, :, slice_idx]
            
            # Random crop
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                
                low_patch = low_slice[top:top+self.patch_size, left:left+self.patch_size]
                full_patch = full_slice[top:top+self.patch_size, left:left+self.patch_size]
            else:
                low_patch = low_slice
                full_patch = full_slice
            
            # Normalize first
            low_patch = self.normalize_hu(low_patch)
            full_patch = self.normalize_hu(full_patch)
            
            # Check if valid
            if self.is_valid_patch(low_patch) and self.is_valid_patch(full_patch):
                return low_patch, full_patch
        
        # If all attempts failed, return the last one anyway
        return low_patch, full_patch
    
    def augment(self, low_patch, full_patch):
        """Enhanced data augmentation"""
        # Horizontal flip
        if random.random() < self.aug_config.get('horizontal_flip', 0.5):
            low_patch = np.fliplr(low_patch).copy()
            full_patch = np.fliplr(full_patch).copy()
        
        # Vertical flip
        if random.random() < self.aug_config.get('vertical_flip', 0.3):
            low_patch = np.flipud(low_patch).copy()
            full_patch = np.flipud(full_patch).copy()
        
        # Rotation
        rotation_range = self.aug_config.get('rotation_range', 0)
        if rotation_range > 0 and random.random() < 0.5:
            angle = random.uniform(-rotation_range, rotation_range)
            low_patch = rotate(low_patch, angle, reshape=False, order=1)
            full_patch = rotate(full_patch, angle, reshape=False, order=1)
        
        # Add small noise to input only
        noise_std = self.aug_config.get('noise_std', 0)
        if noise_std > 0 and random.random() < 0.3:
            noise = np.random.normal(0, noise_std, low_patch.shape).astype(np.float32)
            low_patch = np.clip(low_patch + noise, 0, 1)
        
        return low_patch, full_patch
    
    def __len__(self):
        # More samples per volume for better training
        return len(self.pairs) * 100 if self.mode == 'train' else len(self.pairs) * 10
    
    def __getitem__(self, idx):
        # ========== 개선: On-the-fly 로딩 ==========
        # 필요할 때만 파일 로드 (메모리 효율적)
        
        # Select volume
        vol_idx = idx % len(self.pairs)
        low_file, full_file = self.pairs[vol_idx]
        
        # Load volumes on-the-fly
        low_nii = nib.load(str(low_file))
        full_nii = nib.load(str(full_file))
        
        low_vol = low_nii.get_fdata()
        full_vol = full_nii.get_fdata()
        # ==========================================
        
        # Random crop with validation
        low_patch, full_patch = self.random_crop(low_vol, full_vol)
        
        # Augmentation
        if self.mode == 'train' and self.aug_config:
            low_patch, full_patch = self.augment(low_patch, full_patch)
        
        # To tensor [1, H, W]
        low_tensor = torch.from_numpy(low_patch).unsqueeze(0)
        full_tensor = torch.from_numpy(full_patch).unsqueeze(0)
        
        return low_tensor, full_tensor