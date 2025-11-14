#!/usr/bin/env python3
"""
01_explore_dataset.py
nii_preproc_norm 데이터셋 탐색 및 통계 분석
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def load_nifti(path):
    """NIfTI 파일 로드"""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return arr, spacing


def analyze_volume(nii_path):
    """볼륨 분석"""
    arr, spacing = load_nifti(nii_path)
    
    stats = {
        'path': str(nii_path),
        'shape': arr.shape,
        'spacing': spacing,
        'min': arr.min(),
        'max': arr.max(),
        'mean': arr.mean(),
        'std': arr.std(),
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
    }
    
    return stats, arr


def plot_sample_slices(arr, title, save_path):
    """중간 슬라이스 시각화"""
    num_slices = arr.shape[0]
    indices = [num_slices//4, num_slices//2, 3*num_slices//4]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)
    
    for idx, ax in zip(indices, axes):
        ax.imshow(arr[idx], cmap='gray', vmin=-1, vmax=1)
        ax.set_title(f'Slice {idx}/{num_slices}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_histogram(arr, title, save_path):
    """HU 분포 히스토그램"""
    plt.figure(figsize=(10, 6))
    plt.hist(arr.flatten(), bins=100, range=(-1, 1), alpha=0.7, edgecolor='black')
    plt.title(f'{title} - Pixel Intensity Distribution')
    plt.xlabel('Normalized Intensity')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    # 경로 설정
    data_root = Path(r'E:\LD-CT SR\Data\nii_preproc_norm')
    nc_dir = data_root / 'NC'
    ce_dir = data_root / 'CE'
    
    output_dir = Path(r'E:\LD-CT SR\Data2\samples\data_exploration')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("데이터셋 탐색 시작")
    print("="*80)
    
    # NC 데이터셋 분석
    print("\n[1] NC 데이터셋 분석...")
    nc_patients = sorted([p for p in nc_dir.iterdir() if p.is_dir()])
    print(f"총 환자 수: {len(nc_patients)}")
    
    nc_stats_list = []
    for patient_dir in tqdm(nc_patients[:10], desc="NC 분석"):  # 처음 10명만
        patient_id = patient_dir.name
        nii_path = patient_dir / 'NC_norm.nii.gz'
        
        if not nii_path.exists():
            print(f"Missing: {nii_path}")
            continue
        
        stats, arr = analyze_volume(nii_path)
        stats['patient_id'] = patient_id
        stats['dataset'] = 'NC'
        nc_stats_list.append(stats)
        
        # 첫 번째 환자 시각화
        if len(nc_stats_list) == 1:
            plot_sample_slices(arr, f'NC - Patient {patient_id}', 
                             output_dir / 'nc_sample_slices.png')
            plot_histogram(arr, f'NC - Patient {patient_id}',
                         output_dir / 'nc_histogram.png')
    
    # CE 데이터셋 분석
    print("\n[2] CE 데이터셋 분석...")
    ce_patients = sorted([p for p in ce_dir.iterdir() if p.is_dir()])
    print(f"총 환자 수: {len(ce_patients)}")
    
    ce_stats_list = []
    for patient_dir in tqdm(ce_patients[:10], desc="CE 분석"):  # 처음 10명만
        patient_id = patient_dir.name
        nii_path = patient_dir / 'CE_norm.nii.gz'
        
        if not nii_path.exists():
            print(f"Missing: {nii_path}")
            continue
        
        stats, arr = analyze_volume(nii_path)
        stats['patient_id'] = patient_id
        stats['dataset'] = 'CE'
        ce_stats_list.append(stats)
        
        # 첫 번째 환자 시각화
        if len(ce_stats_list) == 1:
            plot_sample_slices(arr, f'CE - Patient {patient_id}', 
                             output_dir / 'ce_sample_slices.png')
            plot_histogram(arr, f'CE - Patient {patient_id}',
                         output_dir / 'ce_histogram.png')
    
    # 통계 요약
    print("\n[3] 통계 요약 생성...")
    all_stats = nc_stats_list + ce_stats_list
    df = pd.DataFrame(all_stats)
    
    # 숫자형 컬럼만 선택
    numeric_cols = ['min', 'max', 'mean', 'std', 'median', 'q25', 'q75']
    
    print("\n" + "="*80)
    print("NC 데이터셋 통계:")
    print("="*80)
    nc_df = df[df['dataset'] == 'NC'][numeric_cols]
    print(nc_df.describe())
    
    print("\n" + "="*80)
    print("CE 데이터셋 통계:")
    print("="*80)
    ce_df = df[df['dataset'] == 'CE'][numeric_cols]
    print(ce_df.describe())
    
    # CSV 저장
    df.to_csv(output_dir / 'dataset_statistics.csv', index=False)
    
    # 비교 플롯
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean 비교
    axes[0, 0].boxplot([nc_df['mean'], ce_df['mean']], labels=['NC', 'CE'])
    axes[0, 0].set_title('Mean Intensity')
    axes[0, 0].set_ylabel('Normalized Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std 비교
    axes[0, 1].boxplot([nc_df['std'], ce_df['std']], labels=['NC', 'CE'])
    axes[0, 1].set_title('Standard Deviation')
    axes[0, 1].set_ylabel('Normalized Intensity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Min-Max 비교
    axes[1, 0].boxplot([nc_df['min'], ce_df['min']], labels=['NC', 'CE'])
    axes[1, 0].set_title('Minimum Intensity')
    axes[1, 0].set_ylabel('Normalized Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].boxplot([nc_df['max'], ce_df['max']], labels=['NC', 'CE'])
    axes[1, 1].set_title('Maximum Intensity')
    axes[1, 1].set_ylabel('Normalized Intensity')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*80)
    print("분석 완료!")
    print(f"결과 저장: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()