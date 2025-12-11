"""
데이터셋 검증 스크립트
- NIfTI 파일들이 제대로 로드되는지
- HU 값 범위가 정상인지
- Window/Normalize 과정에서 문제가 없는지
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt

print("="*80)
print("데이터셋 검증")
print("="*80)

nc_ct_dir = Path(r"E:\LD-CT SR\Data\Image_NC-CT_v2")
hu_window = (-160, 240)

# NIfTI 파일 목록
files = sorted(list(nc_ct_dir.glob("*.nii.gz")) + list(nc_ct_dir.glob("*.nii")))

print(f"\n✅ 발견된 NIfTI 파일: {len(files)}개\n")

if len(files) == 0:
    print("❌ ERROR: No NIfTI files found!")
    print(f"   Directory: {nc_ct_dir}")
    exit(1)

# 처음 5개 파일 상세 검증
for i, fpath in enumerate(files[:5]):
    print(f"\n{'='*80}")
    print(f"파일 {i+1}: {fpath.name}")
    print(f"{'='*80}")
    
    try:
        # Load NIfTI
        nii = nib.load(str(fpath))
        vol = nii.get_fdata()
        
        print(f"✅ Shape: {vol.shape}")
        print(f"✅ Data type: {vol.dtype}")
        print(f"✅ Spacing: {nii.header.get_zooms()}")
        
        # HU 값 범위
        print(f"\n📊 HU 값 통계:")
        print(f"   Min: {vol.min():.1f}")
        print(f"   Max: {vol.max():.1f}")
        print(f"   Mean: {vol.mean():.1f}")
        print(f"   Std: {vol.std():.1f}")
        
        # Body mask 체크 (간단히)
        center_slice_idx = vol.shape[2] // 2
        center_slice = vol[:, :, center_slice_idx]
        
        body_mask = (center_slice > -500) & (center_slice < 500)
        body_pixels = body_mask.sum()
        total_pixels = center_slice.size
        
        print(f"\n🔍 Center slice (z={center_slice_idx}):")
        print(f"   Min: {center_slice.min():.1f}")
        print(f"   Max: {center_slice.max():.1f}")
        print(f"   Mean: {center_slice.mean():.1f}")
        print(f"   Body pixels: {body_pixels} / {total_pixels} ({body_pixels/total_pixels*100:.1f}%)")
        
        # Windowing + Normalization 테스트
        hu_min, hu_max = hu_window
        slice_clipped = np.clip(center_slice, hu_min, hu_max)
        slice_norm = (slice_clipped - hu_min) / (hu_max - hu_min)
        
        print(f"\n🔧 Window & Normalize 후:")
        print(f"   Min: {slice_norm.min():.3f}")
        print(f"   Max: {slice_norm.max():.3f}")
        print(f"   Mean: {slice_norm.mean():.3f}")
        print(f"   Std: {slice_norm.std():.3f}")
        
        # 문제 진단
        if slice_norm.max() - slice_norm.min() < 0.01:
            print(f"   ⚠️  WARNING: 값의 range가 너무 작음! ({slice_norm.max() - slice_norm.min():.4f})")
        
        if slice_norm.mean() < 0.1 or slice_norm.mean() > 0.9:
            print(f"   ⚠️  WARNING: Mean이 극단적임! ({slice_norm.mean():.3f})")
        
        # 완전히 0인 slice가 있는지 체크
        zero_slices = 0
        for z in range(vol.shape[2]):
            if vol[:, :, z].max() - vol[:, :, z].min() < 1.0:
                zero_slices += 1
        
        if zero_slices > 0:
            print(f"   ⚠️  WARNING: {zero_slices}/{vol.shape[2]} slices have no variation!")
        
        # 시각화 (첫 번째 파일만)
        if i == 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(center_slice, cmap='gray', vmin=-160, vmax=240)
            axes[0].set_title(f'Original HU\n[{center_slice.min():.0f}, {center_slice.max():.0f}]')
            axes[0].axis('off')
            
            axes[1].imshow(slice_clipped, cmap='gray', vmin=hu_min, vmax=hu_max)
            axes[1].set_title(f'After Window\n[{slice_clipped.min():.0f}, {slice_clipped.max():.0f}]')
            axes[1].axis('off')
            
            axes[2].imshow(slice_norm, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title(f'After Normalize\n[{slice_norm.min():.3f}, {slice_norm.max():.3f}]')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig('data_verification.png', dpi=150, bbox_inches='tight')
            print(f"\n📷 시각화 저장: data_verification.png")
        
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")

# 전체 파일 간단 체크
print(f"\n{'='*80}")
print("전체 파일 간단 체크")
print(f"{'='*80}\n")

problem_files = []
for fpath in files:
    try:
        nii = nib.load(str(fpath))
        vol = nii.get_fdata()
        
        # 문제 체크
        if vol.max() - vol.min() < 10:
            problem_files.append((fpath.name, "No variation"))
        elif vol.mean() < -900 or vol.mean() > 900:
            problem_files.append((fpath.name, f"Extreme mean: {vol.mean():.1f}"))
        elif np.isnan(vol).any():
            problem_files.append((fpath.name, "Contains NaN"))
        elif np.isinf(vol).any():
            problem_files.append((fpath.name, "Contains Inf"))
    
    except Exception as e:
        problem_files.append((fpath.name, f"Load error: {e}"))

if problem_files:
    print(f"⚠️  발견된 문제 파일: {len(problem_files)}개\n")
    for fname, issue in problem_files[:10]:  # 처음 10개만 표시
        print(f"   {fname}: {issue}")
    if len(problem_files) > 10:
        print(f"   ... and {len(problem_files) - 10} more")
else:
    print("✅ 모든 파일 정상!")

print(f"\n{'='*80}")
print("검증 완료")
print(f"{'='*80}")