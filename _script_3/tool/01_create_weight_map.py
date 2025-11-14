#!/usr/bin/env python3
"""
01_create_weight_map.py
NC Intensity 기반 Adaptive Weight Map 생성

핵심 아이디어:
- 간 영역 내 어두운 부분 = 종양 (낮은 weight로 회피)
- 간 영역 내 밝은 부분 = 정상 간 (높은 weight로 조영 강화)
- CE 데이터 불필요! NC만으로 해결

데이터 구조:
Input:
  - NC: nii_preproc_norm/NC/{patient_id}/NC_norm.nii.gz
  - Seg: segmentation/{patient_id}/Aorta_seg.nii.gz, Liver_seg.nii.gz

Output:
  - Weight: weight_maps/{patient_id}/NC_weight_map.nii.gz
  - Sample: samples/weight_maps/{patient_id}_weight_map.png
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_adaptive_weight_map(nc_arr, aorta_arr, liver_arr, method='percentile'):
    """
    NC intensity 기반 adaptive weight map 생성
    
    Args:
        nc_arr: NC volume [D, H, W], values in [0, 1]
        aorta_arr: Aorta segmentation [D, H, W], binary
        liver_arr: Liver segmentation [D, H, W], binary
        method: 'percentile', 'threshold', or 'continuous'
    
    Returns:
        weight_map: [D, H, W], values in [0.1, 1.0]
    """
    weight_map = np.ones_like(nc_arr) * 0.1  # Background: 0.1
    
    # 1. Aorta: 항상 1.0 (최고 우선순위)
    weight_map[aorta_arr == 1] = 1.0
    
    # 2. Liver: NC intensity 기반 adaptive weight
    liver_mask = liver_arr == 1
    liver_pixels = nc_arr[liver_mask]
    
    if len(liver_pixels) > 0:
        if method == 'percentile':
            # Percentile 기반 (추천)
            p25 = np.percentile(liver_pixels, 25)
            p50 = np.percentile(liver_pixels, 50)
            p75 = np.percentile(liver_pixels, 75)
            
            # 간 영역에 weight 할당
            liver_intensity = nc_arr[liver_mask]
            liver_weights = np.zeros_like(liver_intensity)
            
            # 어두운 부분 (하위 25%) = 종양 의심 = 낮은 weight
            dark_mask = liver_intensity < p25
            liver_weights[dark_mask] = 0.3
            
            # 중간 밝기 (25-75%)
            medium_mask = (liver_intensity >= p25) & (liver_intensity < p75)
            liver_weights[medium_mask] = 0.6
            
            # 밝은 부분 (상위 25%) = 정상 간 = 높은 weight
            bright_mask = liver_intensity >= p75
            liver_weights[bright_mask] = 0.85
            
            # Weight map에 적용
            weight_map[liver_mask] = liver_weights
            
        elif method == 'threshold':
            # 임계값 기반
            liver_mean = liver_pixels.mean()
            liver_std = liver_pixels.std()
            
            liver_intensity = nc_arr[liver_mask]
            liver_weights = np.zeros_like(liver_intensity)
            
            # 매우 어두움 (mean - 1*std 이하) = 종양
            very_dark = liver_intensity < (liver_mean - 1.0 * liver_std)
            liver_weights[very_dark] = 0.3
            
            # 어두움 (mean - 0.5*std ~ mean)
            dark = (liver_intensity >= (liver_mean - 1.0 * liver_std)) & \
                   (liver_intensity < liver_mean)
            liver_weights[dark] = 0.6
            
            # 밝음 (mean 이상)
            bright = liver_intensity >= liver_mean
            liver_weights[bright] = 0.85
            
            weight_map[liver_mask] = liver_weights
            
        elif method == 'continuous':
            # Continuous mapping (부드러운 전환)
            liver_min = liver_pixels.min()
            liver_max = liver_pixels.max()
            
            liver_intensity = nc_arr[liver_mask]
            
            # Linear mapping: [liver_min, liver_max] → [0.3, 0.85]
            normalized = (liver_intensity - liver_min) / (liver_max - liver_min + 1e-8)
            liver_weights = 0.3 + 0.55 * normalized  # [0.3, 0.85]
            
            weight_map[liver_mask] = liver_weights
    
    return weight_map


def analyze_liver_intensity(nc_arr, liver_arr, patient_id):
    """간 영역 밝기 분석 및 통계"""
    liver_mask = liver_arr == 1
    liver_pixels = nc_arr[liver_mask]
    
    if len(liver_pixels) == 0:
        return None
    
    stats = {
        'patient_id': patient_id,
        'num_pixels': len(liver_pixels),
        'mean': float(liver_pixels.mean()),
        'std': float(liver_pixels.std()),
        'min': float(liver_pixels.min()),
        'max': float(liver_pixels.max()),
        'p25': float(np.percentile(liver_pixels, 25)),
        'p50': float(np.percentile(liver_pixels, 50)),
        'p75': float(np.percentile(liver_pixels, 75)),
    }
    
    return stats


def visualize_weight_map(nc_arr, aorta_arr, liver_arr, weight_map, 
                         output_path, patient_id, num_samples=4):
    """
    Weight map 시각화
    
    Layout:
    Row 1: NC (원본)
    Row 2: Segmentation (Aorta + Liver)
    Row 3: NC Intensity in Liver (히트맵)
    Row 4: Final Weight Map
    """
    num_slices = nc_arr.shape[0]
    
    # 간이 있는 슬라이스 찾기
    valid_slices = []
    for i in range(num_slices):
        if liver_arr[i].sum() > 1000:
            valid_slices.append(i)
    
    if len(valid_slices) < num_samples:
        valid_slices = np.linspace(
            int(num_slices * 0.3),
            int(num_slices * 0.7),
            num_samples,
            dtype=int
        ).tolist()
    else:
        # 균등 샘플링
        step = len(valid_slices) // num_samples
        valid_slices = valid_slices[::step][:num_samples]
    
    # Plot
    fig, axes = plt.subplots(4, num_samples, figsize=(5*num_samples, 20))
    fig.suptitle(
        f'Patient {patient_id} - Adaptive Weight Map\n'
        f'💡 Dark liver region (tumor) = Low weight | Bright region (normal) = High weight',
        fontsize=16, fontweight='bold'
    )
    
    # Custom colormap for weight
    colors = ['blue', 'cyan', 'yellow', 'orange', 'red']
    n_bins = 100
    cmap_weight = LinearSegmentedColormap.from_list('weight', colors, N=n_bins)
    
    for col, slice_idx in enumerate(valid_slices):
        nc_slice = nc_arr[slice_idx]
        aorta_slice = aorta_arr[slice_idx]
        liver_slice = liver_arr[slice_idx]
        weight_slice = weight_map[slice_idx]
        
        # Row 1: Original NC
        axes[0, col].imshow(nc_slice, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'Slice {slice_idx}\nNC Original', fontsize=11, fontweight='bold')
        axes[0, col].axis('off')
        
        # Row 2: Segmentation
        axes[1, col].imshow(nc_slice, cmap='gray', vmin=0, vmax=1)
        if aorta_slice.sum() > 0:
            aorta_mask = np.ma.masked_where(aorta_slice == 0, aorta_slice)
            axes[1, col].imshow(aorta_mask, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
        if liver_slice.sum() > 0:
            liver_mask = np.ma.masked_where(liver_slice == 0, liver_slice)
            axes[1, col].imshow(liver_mask, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
        axes[1, col].set_title(f'Segmentation\nAorta + Liver', fontsize=11, color='green')
        axes[1, col].axis('off')
        
        # Row 3: NC Intensity in Liver (히트맵)
        liver_intensity = np.where(liver_slice == 1, nc_slice, np.nan)
        im3 = axes[2, col].imshow(liver_intensity, cmap='hot', vmin=0, vmax=1)
        axes[2, col].set_title(f'Liver Intensity\n(Dark = Tumor?)', fontsize=11, color='purple')
        axes[2, col].axis('off')
        
        # Row 4: Weight Map
        im4 = axes[3, col].imshow(weight_slice, cmap=cmap_weight, vmin=0.1, vmax=1.0)
        axes[3, col].set_title(f'Weight Map\n(min={weight_slice.min():.2f}, max={weight_slice.max():.2f})', 
                              fontsize=11, color='red', fontweight='bold')
        axes[3, col].axis('off')
    
    # Colorbars
    fig.colorbar(im3, ax=axes[2, :], orientation='horizontal', 
                 fraction=0.046, pad=0.04, label='NC Intensity')
    fig.colorbar(im4, ax=axes[3, :], orientation='horizontal', 
                 fraction=0.046, pad=0.04, label='Weight (Low → High)')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def process_patient(patient_id, nc_dir, seg_dir, output_dir, sample_dir, 
                   visualize=True, method='percentile'):
    """
    환자 1명 처리
    
    Returns:
        success: bool
        message: str
        stats: dict or None
    """
    # 경로
    nc_path = nc_dir / patient_id / 'NC_norm.nii.gz'
    aorta_path = seg_dir / patient_id / 'Aorta_seg.nii.gz'
    liver_path = seg_dir / patient_id / 'Liver_seg.nii.gz'
    
    # 존재 확인
    if not nc_path.exists():
        return False, "NC_norm.nii.gz not found", None
    if not aorta_path.exists():
        return False, "Aorta_seg.nii.gz not found", None
    if not liver_path.exists():
        return False, "Liver_seg.nii.gz not found", None
    
    # 로드
    nc_img = sitk.ReadImage(str(nc_path))
    nc_arr = sitk.GetArrayFromImage(nc_img)
    
    aorta_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(aorta_path)))
    liver_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(liver_path)))
    
    # Shape 확인
    if nc_arr.shape != aorta_arr.shape or nc_arr.shape != liver_arr.shape:
        return False, f"Shape mismatch: NC{nc_arr.shape} vs Seg{aorta_arr.shape}", None
    
    # Weight map 생성
    weight_map = create_adaptive_weight_map(nc_arr, aorta_arr, liver_arr, method=method)
    
    # 저장
    output_path = output_dir / patient_id / 'NC_weight_map.nii.gz'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    weight_img = sitk.GetImageFromArray(weight_map)
    weight_img.CopyInformation(nc_img)
    sitk.WriteImage(weight_img, str(output_path))
    
    # 간 intensity 통계
    stats = analyze_liver_intensity(nc_arr, liver_arr, patient_id)
    
    # 시각화
    if visualize and sample_dir is not None:
        try:
            visualize_weight_map(
                nc_arr=nc_arr,
                aorta_arr=aorta_arr,
                liver_arr=liver_arr,
                weight_map=weight_map,
                output_path=sample_dir / f'{patient_id}_weight_map.png',
                patient_id=patient_id
            )
        except Exception as e:
            return True, f"Weight map OK, but visualization failed: {e}", stats
    
    return True, "Success", stats


def main():
    parser = argparse.ArgumentParser(
        description='Create Adaptive Weight Map from NC Intensity'
    )
    
    parser.add_argument('--nc-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC',
                       help='NC 데이터 경로')
    parser.add_argument('--seg-dir', type=str,
                       default=r'E:\LD-CT SR\Data\segmentation',
                       help='Segmentation 경로')
    parser.add_argument('--output-dir', type=str,
                       default=r'E:\LD-CT SR\Data\weight_maps',
                       help='Weight map 출력 경로')
    parser.add_argument('--sample-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\samples\weight_maps',
                       help='시각화 샘플 경로')
    parser.add_argument('--method', type=str, default='percentile',
                       choices=['percentile', 'threshold', 'continuous'],
                       help='Weight 계산 방법')
    parser.add_argument('--visualize-samples', type=int, default=10)
    parser.add_argument('--start-from', type=int, default=0)
    
    args = parser.parse_args()
    
    nc_dir = Path(args.nc_dir)
    seg_dir = Path(args.seg_dir)
    output_dir = Path(args.output_dir)
    sample_dir = Path(args.sample_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Adaptive Weight Map 생성 (NC Intensity 기반)")
    print("="*80)
    print(f"NC: {nc_dir}")
    print(f"Segmentation: {seg_dir}")
    print(f"출력: {output_dir}")
    print(f"샘플: {sample_dir}")
    print(f"\nMethod: {args.method}")
    print("\n💡 핵심 아이디어:")
    print("  - 간 영역 내 어두운 부분 (종양) = 낮은 weight → 조영 약하게")
    print("  - 간 영역 내 밝은 부분 (정상) = 높은 weight → 조영 강하게")
    print("  - 결과: 정상 간만 밝아지고 종양은 상대적으로 어두워짐!")
    print("="*80)
    
    # 환자 목록
    patient_dirs = sorted([p for p in nc_dir.iterdir() if p.is_dir()])
    total_patients = len(patient_dirs)
    
    if args.start_from > 0:
        patient_dirs = patient_dirs[args.start_from:]
    
    print(f"\n총 환자: {total_patients}")
    print(f"처리할 환자: {len(patient_dirs)}")
    
    success_count = 0
    fail_count = 0
    visualize_count = 0
    all_stats = []
    
    pbar = tqdm(patient_dirs, desc="Processing")
    for patient_dir in pbar:
        patient_id = patient_dir.name
        
        # 이미 처리됨?
        output_path = output_dir / patient_id / 'NC_weight_map.nii.gz'
        if output_path.exists():
            success_count += 1
            pbar.set_postfix({'success': success_count, 'fail': fail_count, 'status': 'skip'})
            continue
        
        # 처리
        should_visualize = visualize_count < args.visualize_samples
        
        success, message, stats = process_patient(
            patient_id=patient_id,
            nc_dir=nc_dir,
            seg_dir=seg_dir,
            output_dir=output_dir,
            sample_dir=sample_dir if should_visualize else None,
            visualize=should_visualize,
            method=args.method
        )
        
        if success:
            success_count += 1
            if should_visualize:
                visualize_count += 1
            if stats is not None:
                all_stats.append(stats)
            pbar.set_postfix({'success': success_count, 'fail': fail_count, 'status': 'ok'})
        else:
            fail_count += 1
            tqdm.write(f"✗ {patient_id}: {message}")
            pbar.set_postfix({'success': success_count, 'fail': fail_count, 'status': 'fail'})
    
    # 결과
    print("\n" + "="*80)
    print("완료!")
    print("="*80)
    print(f"성공: {success_count}/{total_patients}")
    print(f"실패: {fail_count}/{total_patients}")
    print(f"시각화: {visualize_count}개")
    
    # 간 intensity 통계 요약
    if all_stats:
        print("\n간 영역 밝기 통계:")
        means = [s['mean'] for s in all_stats]
        stds = [s['std'] for s in all_stats]
        p25s = [s['p25'] for s in all_stats]
        p75s = [s['p75'] for s in all_stats]
        
        print(f"  평균 밝기: {np.mean(means):.4f} ± {np.std(means):.4f}")
        print(f"  평균 P25: {np.mean(p25s):.4f}")
        print(f"  평균 P75: {np.mean(p75s):.4f}")
        print(f"  → 어두운 영역 (P25 이하) = 종양 의심 → weight 0.3")
        print(f"  → 밝은 영역 (P75 이상) = 정상 간 → weight 0.85")
    
    print(f"\n출력: {output_dir}")
    print(f"샘플: {sample_dir}")
    print("="*80)


if __name__ == '__main__':
    main()