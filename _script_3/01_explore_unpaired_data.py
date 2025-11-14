#!/usr/bin/env python3
"""
01_explore_unpaired_data.py
NC-CT vs CE-CT Unpaired 데이터 분석

데이터 구조:
- NC: E:/LD-CT SR/Data/nii_preproc_norm/NC/{patient_id}/NC_norm.nii.gz
- CE: E:/LD-CT SR/Data/nii_preproc_norm/CE/{patient_id}/CE_norm.nii.gz

핵심 목표:
★ NC의 구조는 절대 보존! (최우선)
★ CE의 조영 효과만 학습
★ Unpaired 데이터로 학습
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse


def load_nifti(path):
    """NIfTI 파일 로드"""
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return arr, spacing


def analyze_volume(nii_path):
    """볼륨 통계 분석"""
    arr, spacing = load_nifti(nii_path)
    
    stats = {
        'path': str(nii_path),
        'num_slices': arr.shape[0],
        'height': arr.shape[1],
        'width': arr.shape[2],
        'spacing_z': spacing[2],
        'spacing_xy': spacing[0],
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
    }
    
    return stats, arr


def extract_random_slices(arr, num_samples=4):
    """볼륨에서 랜덤 슬라이스 추출"""
    num_slices = arr.shape[0]
    
    # 상하 10% 제외
    valid_start = int(num_slices * 0.1)
    valid_end = int(num_slices * 0.9)
    
    indices = np.random.choice(
        range(valid_start, valid_end),
        size=min(num_samples, valid_end - valid_start),
        replace=False
    )
    
    return [arr[i] for i in sorted(indices)]


def visualize_domain_comparison(nc_samples, ce_samples, output_dir):
    """NC vs CE 도메인 비교"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = min(len(nc_samples), len(ce_samples), 4)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 10))
    fig.suptitle('Domain Comparison: NC vs CE (Unpaired)\n'
                 '★ Goal: Preserve NC structure + Add CE contrast ★', 
                 fontsize=16, fontweight='bold')
    
    for i in range(num_samples):
        # NC (구조 기준)
        axes[0, i].imshow(nc_samples[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'NC (Structure Preserve!)\nSample {i+1}', 
                           fontsize=10, fontweight='bold', color='blue')
        axes[0, i].axis('off')
        
        # CE (조영 기준)
        axes[1, i].imshow(ce_samples[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'CE (Contrast Reference)\nSample {i+1}', 
                           fontsize=10, color='red')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'domain_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()


def analyze_contrast_difference(nc_samples, ce_samples, output_dir):
    """조영 효과 차이 분석"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Contrast Enhancement Analysis: CE has brighter vessels/organs', 
                 fontsize=16, fontweight='bold')
    
    num_samples = min(4, len(nc_samples), len(ce_samples))
    
    for i in range(num_samples):
        nc = nc_samples[i]
        ce = ce_samples[i]
        
        # NC
        axes[0, i].imshow(nc, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'NC (No Contrast)', fontweight='bold', color='blue')
        axes[0, i].axis('off')
        
        # CE
        axes[1, i].imshow(ce, cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'CE (With Contrast)', fontweight='bold', color='red')
        axes[1, i].axis('off')
        
        # Histogram 비교
        axes[2, i].hist(nc.flatten(), bins=50, alpha=0.6, label='NC', color='blue', edgecolor='black')
        axes[2, i].hist(ce.flatten(), bins=50, alpha=0.6, label='CE', color='red', edgecolor='black')
        axes[2, i].set_title(f'Intensity Distribution')
        axes[2, i].set_xlabel('Intensity')
        axes[2, i].set_ylabel('Frequency')
        axes[2, i].legend()
        axes[2, i].grid(True, alpha=0.3)
        
        # CE가 더 밝은 영역 (조영 효과)
        axes[2, i].axvline(x=0.7, color='red', linestyle='--', alpha=0.5, label='Contrast Effect')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'contrast_analysis.png', dpi=100, bbox_inches='tight')
    plt.close()


def visualize_statistics(nc_stats_df, ce_stats_df, output_dir):
    """통계 비교 시각화"""
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NC vs CE: Statistical Comparison (Unpaired)', 
                 fontsize=16, fontweight='bold')
    
    # Mean
    axes[0, 0].boxplot([nc_stats_df['mean'], ce_stats_df['mean']], 
                       labels=['NC', 'CE'])
    axes[0, 0].set_title('Mean Intensity', fontweight='bold')
    axes[0, 0].set_ylabel('Normalized Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std
    axes[0, 1].boxplot([nc_stats_df['std'], ce_stats_df['std']], 
                       labels=['NC', 'CE'])
    axes[0, 1].set_title('Standard Deviation', fontweight='bold')
    axes[0, 1].set_ylabel('Normalized Intensity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Max (조영 효과!)
    axes[0, 2].boxplot([nc_stats_df['max'], ce_stats_df['max']], 
                       labels=['NC', 'CE'])
    axes[0, 2].set_title('Max Intensity (Contrast Effect)', fontweight='bold', color='red')
    axes[0, 2].set_ylabel('Normalized Intensity')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Z-axis (unpaired 증거)
    axes[1, 0].boxplot([nc_stats_df['num_slices'], ce_stats_df['num_slices']], 
                       labels=['NC', 'CE'])
    axes[1, 0].set_title('Z-axis Range (Unpaired!)', fontweight='bold')
    axes[1, 0].set_ylabel('Slice Count')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Height
    axes[1, 1].boxplot([nc_stats_df['height'], ce_stats_df['height']], 
                       labels=['NC', 'CE'])
    axes[1, 1].set_title('Image Height', fontweight='bold')
    axes[1, 1].set_ylabel('Pixels')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Width
    axes[1, 2].boxplot([nc_stats_df['width'], ce_stats_df['width']], 
                       labels=['NC', 'CE'])
    axes[1, 2].set_title('Image Width', fontweight='bold')
    axes[1, 2].set_ylabel('Pixels')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistics_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nc-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC')
    parser.add_argument('--ce-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\CE')
    parser.add_argument('--output-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\samples\unpaired_analysis')
    parser.add_argument('--num-patients', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    nc_dir = Path(args.nc_dir)
    ce_dir = Path(args.ce_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Unpaired NC-CE 데이터 분석")
    print("="*80)
    print(f"NC: {nc_dir}")
    print(f"CE: {ce_dir}")
    print(f"출력: {output_dir}")
    print("\n★ 핵심 목표:")
    print("  1. NC 구조 보존 (최우선!)")
    print("  2. CE 조영 효과만 학습")
    print("  3. Unpaired 데이터로 학습")
    print("="*80)
    
    # NC 분석
    print("\n[1] NC 데이터 분석...")
    nc_patients = sorted([p for p in nc_dir.iterdir() if p.is_dir()])
    print(f"전체: {len(nc_patients)}명, 분석: {args.num_patients}명")
    
    nc_stats_list = []
    nc_all_samples = []
    
    for patient_dir in tqdm(nc_patients[:args.num_patients], desc="NC"):
        nii_path = patient_dir / 'NC_norm.nii.gz'
        if not nii_path.exists():
            continue
        
        stats, arr = analyze_volume(nii_path)
        stats['patient_id'] = patient_dir.name
        stats['dataset'] = 'NC'
        nc_stats_list.append(stats)
        
        nc_all_samples.extend(extract_random_slices(arr, 2))
    
    # CE 분석
    print("\n[2] CE 데이터 분석...")
    ce_patients = sorted([p for p in ce_dir.iterdir() if p.is_dir()])
    print(f"전체: {len(ce_patients)}명, 분석: {args.num_patients}명")
    
    ce_stats_list = []
    ce_all_samples = []
    
    for patient_dir in tqdm(ce_patients[:args.num_patients], desc="CE"):
        nii_path = patient_dir / 'CE_norm.nii.gz'
        if not nii_path.exists():
            continue
        
        stats, arr = analyze_volume(nii_path)
        stats['patient_id'] = patient_dir.name
        stats['dataset'] = 'CE'
        ce_stats_list.append(stats)
        
        ce_all_samples.extend(extract_random_slices(arr, 2))
    
    # DataFrame
    nc_df = pd.DataFrame(nc_stats_list)
    ce_df = pd.DataFrame(ce_stats_list)
    
    # 통계 출력
    print("\n" + "="*80)
    print("NC 통계")
    print("="*80)
    print(nc_df[['num_slices', 'mean', 'std', 'max']].describe())
    
    print("\n" + "="*80)
    print("CE 통계")
    print("="*80)
    print(ce_df[['num_slices', 'mean', 'std', 'max']].describe())
    
    # Z축 차이 (unpaired 증거)
    print("\n" + "="*80)
    print("Unpaired 검증: Z축 범위 차이")
    print("="*80)
    print(f"NC 평균 슬라이스: {nc_df['num_slices'].mean():.1f}")
    print(f"CE 평균 슬라이스: {ce_df['num_slices'].mean():.1f}")
    print(f"→ 차이: {abs(nc_df['num_slices'].mean() - ce_df['num_slices'].mean()):.1f}")
    
    # CSV 저장
    nc_df.to_csv(output_dir / 'nc_stats.csv', index=False)
    ce_df.to_csv(output_dir / 'ce_stats.csv', index=False)
    
    # 시각화
    print("\n[3] 시각화...")
    visualize_domain_comparison(nc_all_samples[:4], ce_all_samples[:4], output_dir)
    analyze_contrast_difference(nc_all_samples[:4], ce_all_samples[:4], output_dir)
    visualize_statistics(nc_df, ce_df, output_dir)
    
    print("\n" + "="*80)
    print("완료!")
    print(f"→ {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()