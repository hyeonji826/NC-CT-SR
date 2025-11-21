"""
Check noise level in NC-CT data
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use pathlib for proper path handling
data_dir = Path(r"E:\LD-CT SR\Data\Image_NC-CT")

# Get first file
files = sorted(list(data_dir.glob("*.nii.gz")))

if len(files) == 0:
    print(f"No files found in {data_dir}")
    exit(1)

print(f"Found {len(files)} files")
print(f"Analyzing first file: {files[0].name}")

# Load
nii = nib.load(str(files[0]))
volume = nii.get_fdata()

print(f"\nVolume shape: {volume.shape}")
print(f"Data type: {volume.dtype}")
print(f"Value range: [{volume.min():.1f}, {volume.max():.1f}] HU")

# Get middle slice
mid_idx = volume.shape[2] // 2
mid_slice = volume[:, :, mid_idx]

print(f"\nMiddle slice (index {mid_idx}):")
print(f"  Mean: {mid_slice.mean():.1f} HU")
print(f"  Std:  {mid_slice.std():.1f} HU")

# Get tissue region (exclude air/bone)
tissue_mask = (mid_slice > -100) & (mid_slice < 100)
tissue_region = mid_slice[tissue_mask]

if len(tissue_region) > 0:
    print(f"\nTissue region only (soft tissue):")
    print(f"  Mean: {tissue_region.mean():.1f} HU")
    print(f"  Std:  {tissue_region.std():.1f} HU")
    
    # Noise estimation (std in homogeneous region)
    print(f"\n  ⭐ Estimated noise level: {tissue_region.std():.1f} HU")
    
    if tissue_region.std() < 15:
        print(f"     → LOW NOISE (clean image)")
    elif tissue_region.std() < 30:
        print(f"     → MODERATE NOISE")
    else:
        print(f"     → HIGH NOISE")

# Extract patches for visualization
h, w = mid_slice.shape
patch_size = 100

# Center patch (usually tissue)
center_patch = mid_slice[h//2-patch_size//2:h//2+patch_size//2,
                        w//2-patch_size//2:w//2+patch_size//2]

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Full slice
axes[0, 0].imshow(mid_slice, cmap='gray', vmin=-160, vmax=240)
axes[0, 0].set_title('Full Slice (Abdominal Window)')
axes[0, 0].axis('off')

# Full slice histogram
axes[0, 1].hist(mid_slice.flatten(), bins=100, range=(-160, 240), alpha=0.7, edgecolor='black')
axes[0, 1].axvline(mid_slice.mean(), color='r', linestyle='--', label=f'Mean: {mid_slice.mean():.1f}')
axes[0, 1].set_xlabel('HU Value')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('HU Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Tissue region only
if len(tissue_region) > 0:
    axes[0, 2].hist(tissue_region.flatten(), bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[0, 2].axvline(tissue_region.mean(), color='r', linestyle='--', label=f'Mean: {tissue_region.mean():.1f}')
    axes[0, 2].set_xlabel('HU Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title(f'Tissue Only (Std: {tissue_region.std():.1f} HU)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

# Center patch
axes[1, 0].imshow(center_patch, cmap='gray', vmin=-160, vmax=240)
axes[1, 0].set_title(f'Center Patch {patch_size}x{patch_size}')
axes[1, 0].axis('off')

# Center patch - enhanced contrast
patch_mean = center_patch.mean()
axes[1, 1].imshow(center_patch, cmap='gray', vmin=patch_mean-50, vmax=patch_mean+50)
axes[1, 1].set_title('Center Patch (Enhanced Contrast)')
axes[1, 1].axis('off')

# Noise pattern (high-pass filter)
from scipy import ndimage
smoothed = ndimage.gaussian_filter(center_patch, sigma=2)
noise_pattern = center_patch - smoothed

axes[1, 2].imshow(noise_pattern, cmap='gray', vmin=-30, vmax=30)
axes[1, 2].set_title(f'Noise Pattern (Std: {noise_pattern.std():.2f})')
axes[1, 2].axis('off')
axes[1, 2].text(0.5, -0.1, f'High-freq component only', 
               transform=axes[1, 2].transAxes, ha='center')

plt.tight_layout()

# Save
output_path = Path(r"E:\LD-CT SR\Outputs") / "noise_analysis.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved visualization: {output_path}")

plt.show()

# Final assessment
print("\n" + "="*80)
print("ASSESSMENT")
print("="*80)

if len(tissue_region) > 0:
    noise_std = tissue_region.std()
    
    print(f"\nNoise level: {noise_std:.1f} HU")
    
    if noise_std < 15:
        print("\n⚠️  WARNING: Very low noise!")
        print("   Your NC-CT data appears to be quite clean.")
        print("   This might be:")
        print("     1. Standard dose CT (not low-dose)")
        print("     2. Already preprocessed data")
        print("     3. High-quality scanner")
        print("\n   → Denoising improvement may be LIMITED")
        
    elif noise_std < 25:
        print("\n✅ Moderate noise level")
        print("   This is typical for low-dose CT")
        print("   Denoising should show improvement")
        
    else:
        print("\n✅ High noise level")
        print("   Significant denoising improvement expected")

# Analyze multiple files
print("\n" + "="*80)
print("ANALYZING MULTIPLE FILES")
print("="*80)

noise_levels = []
for i, file in enumerate(files[:5]):  # First 5 files
    nii = nib.load(str(file))
    vol = nii.get_fdata()
    mid_slice = vol[:, :, vol.shape[2]//2]
    
    tissue_mask = (mid_slice > -100) & (mid_slice < 100)
    tissue = mid_slice[tissue_mask]
    
    if len(tissue) > 0:
        noise = tissue.std()
        noise_levels.append(noise)
        print(f"  File {i+1}: {file.name[:20]:20s} → Noise: {noise:.1f} HU")

if noise_levels:
    avg_noise = np.mean(noise_levels)
    print(f"\n  Average noise across files: {avg_noise:.1f} HU")
    
    if avg_noise < 15:
        print(f"  ⚠️  Overall assessment: LOW NOISE dataset")
        print(f"     → Limited room for improvement")
    elif avg_noise < 25:
        print(f"  ✅ Overall assessment: MODERATE NOISE dataset")
        print(f"     → Good for denoising experiments")
    else:
        print(f"  ✅ Overall assessment: HIGH NOISE dataset")
        print(f"     → Excellent for denoising experiments")

print("\n" + "="*80)