# -*- coding: utf-8 -*-
"""
Pseudo-pair Generation for NC-CE CT
각 NC 슬라이스에 대해 가장 유사한 CE 슬라이스를 찾아 매칭
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
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def compute_histogram_similarity(img1, img2, bins=256):
    """히스토그램 유사도 계산"""
    hist1, _ = np.histogram(img1.flatten(), bins=bins, range=(0, 1))
    hist2, _ = np.histogram(img2.flatten(), bins=bins, range=(0, 1))
    
    # Normalize
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    
    # Chi-square distance
    chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
    
    # Convert to similarity (0-1, higher is better)
    similarity = 1 / (1 + chi_square)
    return similarity


def compute_correlation(img1, img2):
    """픽셀 간 상관관계"""
    corr, _ = pearsonr(img1.flatten(), img2.flatten())
    return corr


def compute_combined_similarity(nc_slice, ce_slice):
    """
    여러 metric 조합한 유사도
    Returns: similarity score (0-1, higher is better)
    """
    # SSIM
    ssim_score = ssim(nc_slice, ce_slice, data_range=1.0)
    
    # Histogram similarity
    hist_score = compute_histogram_similarity(nc_slice, ce_slice)
    
    # Correlation
    corr_score = (compute_correlation(nc_slice, ce_slice) + 1) / 2  # [-1,1] → [0,1]
    
    # Mean Absolute Error (inverted to similarity)
    mae = np.mean(np.abs(nc_slice - ce_slice))
    mae_score = 1 / (1 + mae)
    
    # Weighted combination
    combined_score = (
        0.4 * ssim_score +
        0.3 * hist_score +
        0.2 * corr_score +
        0.1 * mae_score
    )
    
    return combined_score, {
        'ssim': ssim_score,
        'hist': hist_score,
        'corr': corr_score,
        'mae': mae_score
    }


def find_best_match_for_nc_slice(nc_slice, ce_volume_array):
    """
    NC 슬라이스에 대해 CE volume에서 가장 유사한 슬라이스 찾기
    """
    D_ce = ce_volume_array.shape[0]
    best_score = -1
    best_idx = 0
    best_metrics = {}
    
    for ce_idx in range(D_ce):
        ce_slice = ce_volume_array[ce_idx]
        score, metrics = compute_combined_similarity(nc_slice, ce_slice)
        
        if score > best_score:
            best_score = score
            best_idx = ce_idx
            best_metrics = metrics
    
    return best_idx, best_score, best_metrics


def create_pseudo_pairs(args):
    """모든 NC-CE volume pair에 대해 pseudo-pairing 생성"""
    root = Path(args.root)
    
    # Load pairs.csv
    pairs_df = pd.read_csv(root / args.pairs_csv)
    
    all_pseudo_pairs = []
    
    print(f"\n{'='*80}")
    print("Creating Pseudo-pairs for NC-CE CT")
    print(f"{'='*80}\n")
    
    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Processing volumes"):
        pid = row['id7']
        nc_path = Path(row['input_nc_norm'])
        ce_path = Path(row['target_ce_norm'])
        
        if not nc_path.exists() or not ce_path.exists():
            print(f"[SKIP] {pid}: File not found")
            continue
        
        # Load volumes
        nc_img = sitk.ReadImage(str(nc_path))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        ce_img = sitk.ReadImage(str(ce_path))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        
        D_nc, H, W = nc_arr.shape
        D_ce = ce_arr.shape[0]
        
        # 각 NC 슬라이스마다 best CE 슬라이스 찾기
        volume_pairs = []
        
        for nc_idx in range(D_nc):
            nc_slice = nc_arr[nc_idx]
            
            # Find best matching CE slice
            best_ce_idx, score, metrics = find_best_match_for_nc_slice(nc_slice, ce_arr)
            
            # Only keep pairs with reasonable similarity
            if score >= args.min_similarity:
                volume_pairs.append({
                    'patient_id': pid,
                    'nc_path': str(nc_path),
                    'nc_slice_idx': nc_idx,
                    'ce_path': str(ce_path),
                    'ce_slice_idx': best_ce_idx,
                    'similarity_score': score,
                    'ssim': metrics['ssim'],
                    'hist_sim': metrics['hist'],
                    'correlation': metrics['corr'],
                    'mae_sim': metrics['mae']
                })
        
        all_pseudo_pairs.extend(volume_pairs)
        
        print(f"  [{pid}] NC slices: {D_nc}, CE slices: {D_ce}, "
              f"Valid pairs: {len(volume_pairs)} (threshold: {args.min_similarity:.2f})")
    
    # Save to CSV
    output_csv = root / args.output_csv
    df_pseudo = pd.DataFrame(all_pseudo_pairs)
    df_pseudo.to_csv(output_csv, index=False)
    
    # Statistics
    print(f"\n{'='*80}")
    print("Pseudo-pairing Results:")
    print(f"  Total pseudo-pairs: {len(all_pseudo_pairs)}")
    print(f"  Similarity score:")
    print(f"    Mean: {df_pseudo['similarity_score'].mean():.4f}")
    print(f"    Min:  {df_pseudo['similarity_score'].min():.4f}")
    print(f"    Max:  {df_pseudo['similarity_score'].max():.4f}")
    print(f"  Per patient:")
    print(f"    Mean pairs: {len(all_pseudo_pairs) / len(pairs_df):.1f}")
    print(f"  Saved to: {output_csv}")
    print(f"{'='*80}\n")
    
    # Visualize sample pairs
    if args.visualize:
        visualize_sample_pairs(df_pseudo, args.num_visualize, root / 'Data' / 'pseudo_pair_samples')


def visualize_sample_pairs(df_pseudo, num_samples, output_dir):
    """샘플 pseudo-pair 시각화"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample high, medium, low similarity pairs
    high = df_pseudo.nlargest(num_samples // 3, 'similarity_score')
    medium = df_pseudo.iloc[len(df_pseudo) // 2 - num_samples // 6 : len(df_pseudo) // 2 + num_samples // 6]
    low = df_pseudo.nsmallest(num_samples // 3, 'similarity_score')
    
    samples = pd.concat([high, medium, low])
    
    for idx, row in samples.iterrows():
        nc_img = sitk.ReadImage(row['nc_path'])
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        nc_slice = nc_arr[int(row['nc_slice_idx'])]
        
        ce_img = sitk.ReadImage(row['ce_path'])
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        ce_slice = ce_arr[int(row['ce_slice_idx'])]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(nc_slice, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'NC (slice {int(row["nc_slice_idx"])})', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(ce_slice, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'CE (slice {int(row["ce_slice_idx"])})', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        fig.suptitle(f'Patient: {row["patient_id"]} | Similarity: {row["similarity_score"]:.4f}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = output_dir / f'{row["patient_id"]}_nc{int(row["nc_slice_idx"])}_ce{int(row["ce_slice_idx"])}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"  Visualized {len(samples)} sample pairs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Create Pseudo-pairs for NC-CE CT')
    
    parser.add_argument('--root', default=r'E:\LD-CT SR',
                       help='Root directory')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv',
                       help='Original pairs CSV')
    parser.add_argument('--output-csv', default='Data/pseudo_pairs.csv',
                       help='Output pseudo-pairs CSV')
    parser.add_argument('--min-similarity', type=float, default=0.3,
                       help='Minimum similarity threshold (0-1)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample pairs')
    parser.add_argument('--num-visualize', type=int, default=30,
                       help='Number of sample pairs to visualize')
    
    args = parser.parse_args()
    
    create_pseudo_pairs(args)


if __name__ == '__main__':
    main()