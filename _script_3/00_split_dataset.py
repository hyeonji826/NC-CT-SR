#!/usr/bin/env python3
"""
00_split_dataset.py
데이터셋을 Train/Val/Test로 분할

목표:
- Train: 70%
- Val: 15%
- Test: 15%

출력:
- split_info.json (환자 ID 목록)
"""

import json
from pathlib import Path
import argparse
import random
import numpy as np


def split_dataset(nc_dir, output_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    데이터셋을 Train/Val/Test로 분할
    
    Args:
        nc_dir: NC 데이터 디렉토리
        output_path: split_info.json 저장 경로
        train_ratio: Train 비율 (0.7 = 70%)
        val_ratio: Val 비율 (0.15 = 15%)
        seed: Random seed
    """
    nc_base = Path(nc_dir)
    
    # 환자 목록
    patient_dirs = sorted([p for p in nc_base.iterdir() if p.is_dir()])
    patient_ids = [p.name for p in patient_dirs]
    
    total_patients = len(patient_ids)
    print(f"총 환자: {total_patients}명")
    
    # Shuffle
    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(patient_ids)
    
    # Split indices
    train_end = int(total_patients * train_ratio)
    val_end = train_end + int(total_patients * val_ratio)
    
    train_ids = patient_ids[:train_end]
    val_ids = patient_ids[train_end:val_end]
    test_ids = patient_ids[val_end:]
    
    # Split info
    split_info = {
        'seed': seed,
        'total': total_patients,
        'train': {
            'count': len(train_ids),
            'ratio': len(train_ids) / total_patients,
            'ids': train_ids
        },
        'val': {
            'count': len(val_ids),
            'ratio': len(val_ids) / total_patients,
            'ids': val_ids
        },
        'test': {
            'count': len(test_ids),
            'ratio': len(test_ids) / total_patients,
            'ids': test_ids
        }
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Print
    print("\n" + "="*80)
    print("데이터셋 분할 완료!")
    print("="*80)
    print(f"Train: {len(train_ids)}명 ({len(train_ids)/total_patients*100:.1f}%)")
    print(f"Val:   {len(val_ids)}명 ({len(val_ids)/total_patients*100:.1f}%)")
    print(f"Test:  {len(test_ids)}명 ({len(test_ids)/total_patients*100:.1f}%)")
    print(f"\n저장: {output_path}")
    print("="*80)
    
    # 샘플 출력
    print("\n샘플:")
    print(f"Train (처음 5명): {train_ids[:5]}")
    print(f"Val (처음 5명):   {val_ids[:5]}")
    print(f"Test (처음 5명):  {test_ids[:5]}")
    
    return split_info


def main():
    parser = argparse.ArgumentParser(
        description='데이터셋을 Train/Val/Test로 분할'
    )
    
    parser.add_argument('--nc-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC',
                       help='NC 데이터 디렉토리')
    parser.add_argument('--output', type=str,
                       default=r'E:\LD-CT SR\Data\split_info.json',
                       help='출력 JSON 경로')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "Train + Val ratio must be < 1.0"
    
    print(f"분할 비율:")
    print(f"  Train: {args.train_ratio*100:.0f}%")
    print(f"  Val:   {args.val_ratio*100:.0f}%")
    print(f"  Test:  {test_ratio*100:.0f}%")
    print(f"  Seed:  {args.seed}")
    
    split_dataset(
        nc_dir=args.nc_dir,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()