#!/usr/bin/env python3
"""
Noise2Noise Pairing for NC-CT
같은 환자의 NC 슬라이스끼리 페어링하여 self-supervised denoising 데이터 생성
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt


def compute_slice_similarity(slice1, slice2):
    """두 슬라이스 간 구조적 유사도 계산"""
    # 크기가 다르면 리사이즈
    if slice1.shape != slice2.shape:
        from skimage.transform import resize
        slice2 = resize(slice2, slice1.shape, 
                       order=1, preserve_range=True, anti_aliasing=True)
    
    # 정규화
    s1_norm = (slice1 - slice1.min()) / (slice1.max() - slice1.min() + 1e-8)
    s2_norm = (slice2 - slice2.min()) / (slice2.max() - slice2.min() + 1e-8)
    
    # SSIM
    ssim_score = ssim(s1_norm, s2_norm, data_range=1.0)
    
    return ssim_score


def create_noise2noise_pairs(args):
    """NC volume 내에서 Noise2Noise pair 생성"""
    root = Path(args.root)
    pairs_df = pd.read_csv(root / args.pairs_csv)
    
    all_n2n_pairs = []
    
    print(f"\n{'='*80}")
    print("Creating Noise2Noise Pairs for NC-CT")
    print(f"{'='*80}\n")
    
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Processing volumes"):
        pid = row['id7']
        nc_path = Path(row['input_nc_norm'])
        
        if not nc_path.exists():
            print(f"[SKIP] {pid}: File not found")
            continue
        
        # Load NC volume
        nc_img = sitk.ReadImage(str(nc_path))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        D, H, W = nc_arr.shape
        
        # 페어링 전략
        volume_pairs = []
        
        if args.pairing_strategy == 'adjacent':
            # 전략 1: 인접 슬라이스 페어링
            for i in range(D - 1):
                slice1 = nc_arr[i]
                slice2 = nc_arr[i + 1]
                
                # 유사도 체크
                sim = compute_slice_similarity(slice1, slice2)
                
                if sim >= args.min_similarity:
                    volume_pairs.append({
                        'patient_id': pid,
                        'nc_path': str(nc_path),
                        'slice1_idx': i,
                        'slice2_idx': i + 1,
                        'similarity': sim,
                        'strategy': 'adjacent'
                    })
        
        elif args.pairing_strategy == 'skip':
            # 전략 2: 1-2개 슬라이스 건너뛰기 (더 독립적인 노이즈)
            skip_sizes = [2, 3]
            for skip in skip_sizes:
                for i in range(D - skip):
                    slice1 = nc_arr[i]
                    slice2 = nc_arr[i + skip]
                    
                    sim = compute_slice_similarity(slice1, slice2)
                    
                    if sim >= args.min_similarity:
                        volume_pairs.append({
                            'patient_id': pid,
                            'nc_path': str(nc_path),
                            'slice1_idx': i,
                            'slice2_idx': i + skip,
                            'similarity': sim,
                            'strategy': f'skip_{skip}'
                        })
        
        elif args.pairing_strategy == 'mixed':
            # 전략 3: Adjacent + Skip 혼합
            # Adjacent pairs
            for i in range(D - 1):
                slice1 = nc_arr[i]
                slice2 = nc_arr[i + 1]
                sim = compute_slice_similarity(slice1, slice2)
                
                if sim >= args.min_similarity:
                    volume_pairs.append({
                        'patient_id': pid,
                        'nc_path': str(nc_path),
                        'slice1_idx': i,
                        'slice2_idx': i + 1,
                        'similarity': sim,
                        'strategy': 'adjacent'
                    })
            
            # Skip pairs (every 3rd slice for more diversity)
            skip = 3
            for i in range(D - skip):
                slice1 = nc_arr[i]
                slice2 = nc_arr[i + skip]
                sim = compute_slice_similarity(slice1, slice2)
                
                if sim >= args.min_similarity:
                    volume_pairs.append({
                        'patient_id': pid,
                        'nc_path': str(nc_path),
                        'slice1_idx': i,
                        'slice2_idx': i + skip,
                        'similarity': sim,
                        'strategy': f'skip_{skip}'
                    })
        
        all_n2n_pairs.extend(volume_pairs)
        
        print(f"  [{pid}] Total slices: {D}, Valid N2N pairs: {len(volume_pairs)}")
    
    # Save to CSV
    output_csv = root / args.output_csv
    df_n2n = pd.DataFrame(all_n2n_pairs)
    df_n2n.to_csv(output_csv, index=False)
    
    # Statistics
    print(f"\n{'='*80}")
    print("Noise2Noise Pairing Results:")
    print(f"  Total N2N pairs: {len(all_n2n_pairs)}")
    print(f"  Similarity score:")
    print(f"    Mean: {df_n2n['similarity'].mean():.4f}")
    print(f"    Min:  {df_n2n['similarity'].min():.4f}")
    print(f"    Max:  {df_n2n['similarity'].max():.4f}")
    print(f"  Per patient:")
    print(f"    Mean pairs: {len(all_n2n_pairs) / len(pairs_df):.1f}")
    
    if 'strategy' in df_n2n.columns:
        print(f"  By strategy:")
        for strategy, count in df_n2n['strategy'].value_counts().items():
            print(f"    {strategy}: {count}")
    
    print(f"  Saved to: {output_csv}")
    print(f"{'='*80}\n")
    
    # Visualize
    if args.visualize:
        visualize_n2n_pairs(df_n2n, args.num_visualize, 
                           root / 'Data' / 'noise2noise_samples')


def visualize_n2n_pairs(df_n2n, num_samples, output_dir):
    """Noise2Noise pair 시각화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample different similarity ranges
    high = df_n2n.nlargest(num_samples // 3, 'similarity')
    medium = df_n2n.iloc[len(df_n2n) // 2 - num_samples // 6 : 
                         len(df_n2n) // 2 + num_samples // 6]
    low = df_n2n.nsmallest(num_samples // 3, 'similarity')
    
    samples = pd.concat([high, medium, low])
    
    for idx, row in samples.iterrows():
        nc_img = sitk.ReadImage(row['nc_path'])
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        slice1 = nc_arr[int(row['slice1_idx'])]
        slice2 = nc_arr[int(row['slice2_idx'])]
        
        # 크기 맞추기
        if slice1.shape != slice2.shape:
            from skimage.transform import resize
            slice2 = resize(slice2, slice1.shape, 
                          order=1, preserve_range=True, anti_aliasing=True)
        
        # Difference map
        diff = np.abs(slice1 - slice2)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(slice1, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Slice {int(row["slice1_idx"])}', 
                         fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(slice2, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Slice {int(row["slice2_idx"])}', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.2)
        axes[2].set_title('Difference (Noise)', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        strategy = row.get('strategy', 'unknown')
        fig.suptitle(f'Patient: {row["patient_id"]} | Strategy: {strategy} | '
                    f'Similarity: {row["similarity"]:.4f}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = output_dir / f'{row["patient_id"]}_s{int(row["slice1_idx"])}_'
        f's{int(row["slice2_idx"])}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"  Visualized {len(samples)} sample N2N pairs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Create Noise2Noise Pairs for Self-supervised Denoising'
    )
    
    parser.add_argument('--root', default=r'E:\LD-CT SR',
                       help='Root directory')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv',
                       help='Original pairs CSV (for NC paths)')
    parser.add_argument('--output-csv', default='Data/noise2noise_pairs.csv',
                       help='Output Noise2Noise pairs CSV')
    parser.add_argument('--pairing-strategy', type=str, default='mixed',
                       choices=['adjacent', 'skip', 'mixed'],
                       help='Pairing strategy: adjacent, skip, or mixed')
    parser.add_argument('--min-similarity', type=float, default=0.7,
                       help='Minimum SSIM threshold for pairing (0-1)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample pairs')
    parser.add_argument('--num-visualize', type=int, default=30,
                       help='Number of sample pairs to visualize')
    
    args = parser.parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Pairing strategy: {args.pairing_strategy}")
    print(f"  Min similarity: {args.min_similarity}")
    print()
    
    create_noise2noise_pairs(args)


if __name__ == '__main__':
    main()