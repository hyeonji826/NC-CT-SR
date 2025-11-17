# E:\LD-CT SR\_scripts_4_wavelet\debug_data.py

import torch
import numpy as np
from dataset import CTDenoiseDataset

# Dataset 로드
dataset = CTDenoiseDataset(
    low_dose_dir=r"E:\LD-CT SR\Data\Images_low_dose",
    full_dose_dir=r"E:\LD-CT SR\Data\Images_full_dose",
    hu_window=(-160, 240),
    patch_size=128,
    config_aug=None,
    mode='train'
)

print("Testing 10 samples...")

for i in range(10):
    try:
        low, full = dataset[i]
        
        print(f"\nSample {i}:")
        print(f"  Low shape: {low.shape}")
        print(f"  Low range: [{low.min():.4f}, {low.max():.4f}]")
        print(f"  Low mean: {low.mean():.4f}")
        print(f"  Has NaN: {torch.isnan(low).any().item()}")
        print(f"  Has Inf: {torch.isinf(low).any().item()}")
        
        print(f"  Full shape: {full.shape}")
        print(f"  Full range: [{full.min():.4f}, {full.max():.4f}]")
        print(f"  Full mean: {full.mean():.4f}")
        print(f"  Has NaN: {torch.isnan(full).any().item()}")
        print(f"  Has Inf: {torch.isinf(full).any().item()}")
        
        # Check for suspicious values
        if low.min() < -1 or low.max() > 2:
            print("  ⚠️ WARNING: Low values out of expected range!")
        if full.min() < -1 or full.max() > 2:
            print("  ⚠️ WARNING: Full values out of expected range!")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n✅ Data check complete")