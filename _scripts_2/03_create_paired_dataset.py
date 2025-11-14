#!/usr/bin/env python3
"""
03_create_paired_dataset.py
Synthetic LD ↔ Clean NC 페어링 데이터셋 생성

Perfect Pairing:
- 같은 환자, 같은 슬라이스
- 100% alignment 보장
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import argparse
import matplotlib.pyplot as plt


def create_pairs(clean_dir, synthetic_ld_dir, output_csv, sample_dir=None, visualize_samples=5):
    """
    Clean NC ↔ Synthetic LD 페어 생성
    
    Args:
        clean_dir: Clean NC 경로
        synthetic_ld_dir: Synthetic LD 경로
        output_csv: 출력 CSV 경로
        sample_dir: 샘플 저장 경로
        visualize_samples: 시각화할 샘플 수
    """
    clean_dir = Path(clean_dir)
    synthetic_ld_dir = Path(synthetic_ld_dir)
    
    pairs = []
    
    # 환자 목록
    clean_patients = sorted([p for p in clean_dir.iterdir() if p.is_dir()])
    
    print(f"총 환자 수: {len(clean_patients)}")
    
    visualize_count = 0
    
    for patient_dir in tqdm(clean_patients, desc="Creating pairs"):
        patient_id = patient_dir.name
        
        # 경로
        clean_path = patient_dir / 'NC_norm.nii.gz'
        synthetic_path = synthetic_ld_dir / patient_id / 'NC_synthetic_ld.nii.gz'
        
        # 존재 여부 확인
        if not clean_path.exists():
            print(f"Missing clean: {clean_path}")
            continue
        
        if not synthetic_path.exists():
            print(f"Missing synthetic: {synthetic_path}")
            continue
        
        # 볼륨 로드
        clean_img = sitk.ReadImage(str(clean_path))
        clean_arr = sitk.GetArrayFromImage(clean_img)
        
        synthetic_img = sitk.ReadImage(str(synthetic_path))
        synthetic_arr = sitk.GetArrayFromImage(synthetic_img)
        
        # Shape 확인
        if clean_arr.shape != synthetic_arr.shape:
            print(f"Shape mismatch for {patient_id}: {clean_arr.shape} vs {synthetic_arr.shape}")
            continue
        
        num_slices = clean_arr.shape[0]
        
        # 모든 슬라이스 페어링
        for slice_idx in range(num_slices):
            pairs.append({
                'patient_id': patient_id,
                'clean_path': str(clean_path),
                'synthetic_ld_path': str(synthetic_path),
                'slice_idx': slice_idx,
                'num_slices': num_slices
            })
        
        # 시각화 (첫 N명)
        if visualize_count < visualize_samples and sample_dir is not None:
            visualize_patient(clean_arr, synthetic_arr, patient_id, sample_dir)
            visualize_count += 1
    
    # DataFrame 생성
    df = pd.DataFrame(pairs)
    
    print(f"\n생성된 페어 수: {len(df)}")
    print(f"환자 수: {df['patient_id'].nunique()}")
    print(f"평균 슬라이스/환자: {len(df) / df['patient_id'].nunique():.1f}")
    
    # CSV 저장
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\n페어 데이터 저장: {output_csv}")
    
    return df


def visualize_patient(clean_arr, synthetic_arr, patient_id, sample_dir):
    """환자별 페어 시각화"""
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    num_slices = clean_arr.shape[0]
    
    # 5개 슬라이스 샘플링
    slice_indices = np.linspace(0, num_slices-1, 5, dtype=int)
    
    fig, axes = plt.subplots(5, 3, figsize=(15, 25))
    fig.suptitle(f'Patient {patient_id} - Perfect Pairs\n'
                 f'Clean NC ↔ Synthetic LD',
                 fontsize=16, fontweight='bold')
    
    for row, slice_idx in enumerate(slice_indices):
        # Clean NC
        axes[row, 0].imshow(clean_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[row, 0].set_title(f'Slice {slice_idx}: Clean NC (Target)')
        axes[row, 0].axis('off')
        
        # Synthetic LD
        axes[row, 1].imshow(synthetic_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[row, 1].set_title(f'Synthetic LD (Input)')
        axes[row, 1].axis('off')
        
        # Difference
        diff = np.abs(clean_arr[slice_idx] - synthetic_arr[slice_idx])
        im = axes[row, 2].imshow(diff, cmap='hot', vmin=0, vmax=0.15)
        axes[row, 2].set_title(f'Difference (Noise to Remove)')
        axes[row, 2].axis('off')
    
    plt.colorbar(im, ax=axes[:, 2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    save_path = sample_dir / f'{patient_id}_pairs.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def analyze_dataset(df):
    """데이터셋 통계 분석"""
    print("\n" + "="*80)
    print("데이터셋 통계")
    print("="*80)
    
    print(f"\n총 페어 수: {len(df):,}")
    print(f"총 환자 수: {df['patient_id'].nunique()}")
    
    # 환자별 슬라이스 수
    slices_per_patient = df.groupby('patient_id').size()
    print(f"\n환자별 슬라이스 수:")
    print(f"  평균: {slices_per_patient.mean():.1f}")
    print(f"  최소: {slices_per_patient.min()}")
    print(f"  최대: {slices_per_patient.max()}")
    print(f"  중간값: {slices_per_patient.median():.1f}")
    
    # Train/Val 분할 제안
    num_patients = df['patient_id'].nunique()
    train_patients = int(num_patients * 0.8)
    val_patients = num_patients - train_patients
    
    print(f"\n제안 Train/Val 분할 (80/20):")
    print(f"  Train: {train_patients}명 환자")
    print(f"  Val: {val_patients}명 환자")


def split_train_val(df, output_dir, train_ratio=0.8, seed=42):
    """Train/Val 분할"""
    np.random.seed(seed)
    
    # 환자 단위로 분할
    patients = df['patient_id'].unique()
    np.random.shuffle(patients)
    
    split_idx = int(len(patients) * train_ratio)
    train_patients = patients[:split_idx]
    val_patients = patients[split_idx:]
    
    # DataFrame 분할
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    
    # 저장
    output_dir = Path(output_dir)
    train_df.to_csv(output_dir / 'train_pairs.csv', index=False)
    val_df.to_csv(output_dir / 'val_pairs.csv', index=False)
    
    print(f"\nTrain/Val 분할 완료:")
    print(f"  Train: {len(train_df):,} pairs ({len(train_patients)} patients)")
    print(f"  Val: {len(val_df):,} pairs ({len(val_patients)} patients)")
    print(f"  저장: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create Paired Dataset')
    
    parser.add_argument('--clean-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC',
                       help='Clean NC 경로')
    parser.add_argument('--synthetic-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\synthetic_ld\NC',
                       help='Synthetic LD 경로')
    parser.add_argument('--output-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\pairs',
                       help='출력 경로')
    parser.add_argument('--sample-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\samples\pairs',
                       help='샘플 저장 경로')
    parser.add_argument('--visualize-samples', type=int, default=5,
                       help='시각화할 샘플 수')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Train 비율')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Paired Dataset 생성")
    print("="*80)
    print(f"Clean NC: {args.clean_dir}")
    print(f"Synthetic LD: {args.synthetic_dir}")
    print(f"출력: {args.output_dir}")
    print("="*80)
    
    # 페어 생성
    df = create_pairs(
        clean_dir=args.clean_dir,
        synthetic_ld_dir=args.synthetic_dir,
        output_csv=output_dir / 'all_pairs.csv',
        sample_dir=args.sample_dir,
        visualize_samples=args.visualize_samples
    )
    
    # 통계 분석
    analyze_dataset(df)
    
    # Train/Val 분할
    split_train_val(df, output_dir, train_ratio=args.train_ratio)
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80)


if __name__ == '__main__':
    main()