"""
Verify center region cropping on NC-CT data
Rectangle crop: 70% width × 50% height
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
from matplotlib.patches import Rectangle

print("="*80)
print("CENTER REGION CROP VERIFICATION - RECTANGLE")
print("="*80)

# Data directory
data_dir = Path(r"E:\LD-CT SR\Data\Image_NC-CT")
files = sorted(list(data_dir.glob("*.nii.gz")))

if len(files) == 0:
    print(f"❌ No files found in {data_dir}")
    exit(1)

print(f"\n✅ Found {len(files)} files")

# Select 5 random files
random.seed(42)
selected_files = random.sample(files, min(5, len(files)))

print(f"\n📁 Selected files:")
for i, f in enumerate(selected_files):
    print(f"   {i+1}. {f.name}")

# HU window
hu_window = (-160, 240)

# Center region parameters - RECTANGLE
center_ratio_w = 0.55  # Horizontal: 70%
center_ratio_h = 0.6  # Vertical: 50%

print(f"\n🎯 Center region: {center_ratio_w*100:.0f}% × {center_ratio_h*100:.0f}% (Rectangle)")
print(f"   Horizontal (width): {center_ratio_w*100:.0f}%")
print(f"   Vertical (height): {center_ratio_h*100:.0f}%")
print(f"   → Matches abdomen shape (wider horizontally)")

# Create figure
fig, axes = plt.subplots(5, 3, figsize=(15, 20))

for idx, file_path in enumerate(selected_files):
    print(f"\n{'='*80}")
    print(f"Processing: {file_path.name}")
    
    # Load volume
    nii = nib.load(str(file_path))
    volume = nii.get_fdata()
    
    print(f"   Volume shape: {volume.shape}")
    
    # Get middle slice
    mid_idx = volume.shape[2] // 2
    slice_2d = volume[:, :, mid_idx]
    
    h, w = slice_2d.shape
    print(f"   Slice shape: {h} x {w}")
    
    # Calculate center region - RECTANGLE
    margin_h = int(h * (1 - center_ratio_h) / 2)
    margin_w = int(w * (1 - center_ratio_w) / 2)
    
    center_h = h - 2*margin_h
    center_w = w - 2*margin_w
    
    print(f"   Margins: top/bottom={margin_h}px, left/right={margin_w}px")
    print(f"   Center region: {center_h} x {center_w} (Rectangle)")
    
    # Extract center region
    center_slice = slice_2d[margin_h:h-margin_h, margin_w:w-margin_w]
    
    # Measure noise in full vs center
    tissue_mask_full = (slice_2d > -100) & (slice_2d < 100)
    tissue_mask_center = (center_slice > -100) & (center_slice < 100)
    
    if tissue_mask_full.sum() > 100:
        noise_full = slice_2d[tissue_mask_full].std()
    else:
        noise_full = 0
    
    if tissue_mask_center.sum() > 100:
        noise_center = center_slice[tissue_mask_center].std()
    else:
        noise_center = 0
    
    print(f"   Noise (full image): {noise_full:.1f} HU")
    print(f"   Noise (center only): {noise_center:.1f} HU")
    print(f"   Difference: {noise_full - noise_center:+.1f} HU")
    
    # Normalize for display
    slice_norm = np.clip(slice_2d, hu_window[0], hu_window[1])
    slice_norm = (slice_norm - hu_window[0]) / (hu_window[1] - hu_window[0])
    
    center_norm = np.clip(center_slice, hu_window[0], hu_window[1])
    center_norm = (center_norm - hu_window[0]) / (hu_window[1] - hu_window[0])
    
    # Rotate 90 degrees for proper orientation
    slice_norm = np.rot90(slice_norm, k=1)
    center_norm = np.rot90(center_norm, k=1)
    
    # Plot 1: Original with box
    axes[idx, 0].imshow(slice_norm, cmap='gray', vmin=0, vmax=1)
    
    # Draw center region box (adjusted for rotation)
    # After rot90(k=1): coordinates transform
    h_rotated, w_rotated = slice_norm.shape
    rect = Rectangle((margin_h, w_rotated - (margin_w + center_w)), 
                     center_h, center_w,
                     linewidth=3, edgecolor='lime', facecolor='none', linestyle='--')
    axes[idx, 0].add_patch(rect)
    
    axes[idx, 0].set_title(f'{file_path.name[:25]}\nFull Image\nNoise: {noise_full:.1f} HU', 
                          fontsize=10, fontweight='bold')
    axes[idx, 0].axis('off')
    
    # Plot 2: Center region only
    axes[idx, 1].imshow(center_norm, cmap='gray', vmin=0, vmax=1)
    axes[idx, 1].set_title(f'Center {center_ratio_w*100:.0f}%×{center_ratio_h*100:.0f}%\n{center_h}×{center_w}\nNoise: {noise_center:.1f} HU',
                          fontsize=10, fontweight='bold', color='lime')
    axes[idx, 1].axis('off')
    
    # Plot 3: Full image (no box)
    axes[idx, 2].imshow(slice_norm, cmap='gray', vmin=0, vmax=1)
    axes[idx, 2].set_title(f'Full Image\n(for comparison)',
                          fontsize=10)
    axes[idx, 2].axis('off')

plt.suptitle(f'Center Region Crop Verification - RECTANGLE\nCenter: {center_ratio_w*100:.0f}%×{center_ratio_h*100:.0f}% (W×H) | HU Window: {hu_window}',
            fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
output_path = Path(r"E:\LD-CT SR\Outputs") / "crop_verification_rectangle.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')

print(f"\n{'='*80}")
print(f"✅ Verification complete!")
print(f"📊 Saved: {output_path}")
print(f"\n💡 Check the image:")
print(f"   - Left column: Full image with GREEN RECTANGLE showing crop area")
print(f"   - Middle column: CROPPED region only (what model will see)")
print(f"   - Right column: Full image for comparison")
print(f"\n🎯 Verify that:")
print(f"   1. Green box is RECTANGLE (wider horizontally)")
print(f"   2. Box excludes arms and top/bottom background")
print(f"   3. Center region contains only abdomen")
print(f"   4. Noise difference is reasonable")
print("="*80)

plt.show()