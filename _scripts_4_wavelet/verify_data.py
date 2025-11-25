# verify_data.py - NC-CT Noise Distribution Analysis

import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

def analyze_noise_distribution(nc_ct_dir, max_files=50, plot=True):
    """
    Analyze noise distribution in NC-CT dataset
    
    Args:
        nc_ct_dir: Path to NC-CT directory
        max_files: Maximum number of files to scan
        plot: Whether to plot histogram
    """
    nc_ct_dir = Path(nc_ct_dir)
    files = sorted(list(nc_ct_dir.glob("*.nii.gz")))
    
    print(f"\n{'='*80}")
    print(f"NC-CT Noise Distribution Analysis")
    print(f"{'='*80}")
    print(f"Directory: {nc_ct_dir}")
    print(f"Total files: {len(files)}")
    print(f"Scanning: {min(max_files, len(files))} files\n")
    
    noise_samples = []
    edge_samples = []
    file_stats = []
    
    from scipy import ndimage
    
    for file_idx, file_path in enumerate(files[:max_files]):
        try:
            nii = nib.load(str(file_path))
            volume = nii.get_fdata()
            D = volume.shape[2]
            
            # Sample multiple slices per volume
            slice_indices = [D//5, D//3, D//2, 2*D//3, 4*D//5]
            
            file_noises = []
            file_edges = []
            
            for slice_idx in slice_indices:
                if slice_idx >= D:
                    continue
                    
                slice_2d = volume[:, :, slice_idx]
                
                # Center region (exclude arms/equipment)
                h, w = slice_2d.shape
                center_h = slice(h//4, 3*h//4)
                center_w = slice(w//4, 3*w//4)
                center = slice_2d[center_h, center_w]
                
                # Tissue mask (HU -100 ~ 100)
                mask = (center > -100) & (center < 100)
                
                if mask.sum() < 1000:
                    continue
                
                tissue = center[mask]
                noise_std = tissue.std()
                
                # Edge score
                gx = ndimage.sobel(center, axis=0)
                gy = ndimage.sobel(center, axis=1)
                edge_mag = np.hypot(gx, gy)
                edge_score = edge_mag[mask].mean()
                
                noise_samples.append(noise_std)
                edge_samples.append(edge_score)
                file_noises.append(noise_std)
                file_edges.append(edge_score)
            
            if file_noises:
                file_stats.append({
                    'file': file_path.name,
                    'noise_mean': np.mean(file_noises),
                    'noise_min': np.min(file_noises),
                    'noise_max': np.max(file_noises),
                    'edge_mean': np.mean(file_edges)
                })
            
            if (file_idx + 1) % 10 == 0:
                print(f"  Processed {file_idx + 1}/{min(max_files, len(files))} files...")
                
        except Exception as e:
            print(f"  ⚠️ Skip {file_path.name}: {e}")
            continue
    
    if not noise_samples:
        print("❌ No valid samples found!")
        return
    
    noise_samples = np.array(noise_samples)
    edge_samples = np.array(edge_samples)
    
    # Statistics
    print(f"\n{'='*80}")
    print(f"Noise Distribution Statistics (HU)")
    print(f"{'='*80}")
    print(f"Total samples: {len(noise_samples)}")
    print(f"Min:           {noise_samples.min():.1f} HU")
    print(f"5th %ile:      {np.percentile(noise_samples, 5):.1f} HU")
    print(f"10th %ile:     {np.percentile(noise_samples, 10):.1f} HU")
    print(f"25th %ile:     {np.percentile(noise_samples, 25):.1f} HU")
    print(f"Median:        {np.median(noise_samples):.1f} HU")
    print(f"75th %ile:     {np.percentile(noise_samples, 75):.1f} HU")
    print(f"90th %ile:     {np.percentile(noise_samples, 90):.1f} HU")
    print(f"95th %ile:     {np.percentile(noise_samples, 95):.1f} HU")
    print(f"Max:           {noise_samples.max():.1f} HU")
    print(f"Mean ± Std:    {noise_samples.mean():.1f} ± {noise_samples.std():.1f} HU")
    
    # Recommendations
    print(f"\n{'='*80}")
    print(f"HN/LN Selection Recommendations")
    print(f"{'='*80}")
    
    p10 = np.percentile(noise_samples, 10)
    p90 = np.percentile(noise_samples, 90)
    median = np.median(noise_samples)
    
    if p10 > 40:
        print("⚠️  WARNING: Entire dataset has HIGH NOISE!")
        print(f"   10th percentile = {p10:.1f} HU (should be < 30 HU for good LN)")
        print(f"\n   Strategy: Use RELATIVE thresholds")
        print(f"   - HN: > 90th percentile ({p90:.1f} HU)")
        print(f"   - LN: < 20th percentile + high edge")
    elif p10 > 30:
        print("⚠️  Dataset is moderately noisy overall")
        print(f"   10th percentile = {p10:.1f} HU")
        print(f"\n   Strategy: Use lower percentiles")
        print(f"   - HN: > 90th percentile ({p90:.1f} HU)")
        print(f"   - LN: < 15th percentile + high edge")
    else:
        print("✅ Good noise distribution for HN/LN separation")
        print(f"   10th percentile = {p10:.1f} HU (< 30 HU)")
        print(f"\n   Strategy: Use absolute + relative thresholds")
        print(f"   - HN: > 90th percentile ({p90:.1f} HU)")
        print(f"   - LN: < 25 HU + high edge")
    
    # Top LN candidates (low noise + high edge)
    print(f"\n{'='*80}")
    print(f"Top 5 LN Candidates (Low Noise + High Edge)")
    print(f"{'='*80}")
    
    # Below median noise
    low_noise_mask = noise_samples < median
    low_noise_indices = np.where(low_noise_mask)[0]
    
    if len(low_noise_indices) > 0:
        low_noise_edges = edge_samples[low_noise_indices]
        top_edge_in_low_noise = low_noise_indices[np.argsort(low_noise_edges)[-5:]]
        
        for rank, idx in enumerate(reversed(top_edge_in_low_noise), 1):
            print(f"{rank}. Noise: {noise_samples[idx]:.1f} HU, Edge: {edge_samples[idx]:.1f}")
    else:
        print("No samples below median found")
    
    # Top HN candidates
    print(f"\n{'='*80}")
    print(f"Top 5 HN Candidates (High Noise)")
    print(f"{'='*80}")
    
    top_hn_indices = np.argsort(noise_samples)[-5:]
    for rank, idx in enumerate(reversed(top_hn_indices), 1):
        print(f"{rank}. Noise: {noise_samples[idx]:.1f} HU, Edge: {edge_samples[idx]:.1f}")
    
    # Plot histogram
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Noise histogram
        axes[0].hist(noise_samples, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0].axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median: {median:.1f}')
        axes[0].axvline(p10, color='green', linestyle='--', linewidth=2, label=f'10th %: {p10:.1f}')
        axes[0].axvline(p90, color='orange', linestyle='--', linewidth=2, label=f'90th %: {p90:.1f}')
        axes[0].set_xlabel('Noise Std (HU)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('NC-CT Noise Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Noise vs Edge scatter
        axes[1].scatter(noise_samples, edge_samples, alpha=0.5, s=20)
        axes[1].axvline(median, color='red', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].axhline(np.median(edge_samples), color='blue', linestyle='--', linewidth=2, alpha=0.5)
        axes[1].set_xlabel('Noise Std (HU)', fontsize=12)
        axes[1].set_ylabel('Edge Score', fontsize=12)
        axes[1].set_title('Noise vs Edge Complexity', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_path = Path(nc_ct_dir).parent / 'noise_distribution_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved: {output_path}")
        plt.close()
    
    return noise_samples, edge_samples, file_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze NC-CT noise distribution')
    parser.add_argument('--data_dir', type=str, 
                       default=r'E:/LD-CT SR/Data/Image_NC-CT',
                       help='Path to NC-CT directory')
    parser.add_argument('--max_files', type=int, default=50,
                       help='Maximum number of files to scan')
    parser.add_argument('--no_plot', action='store_true',
                       help='Disable plotting')
    
    args = parser.parse_args()
    
    analyze_noise_distribution(
        args.data_dir,
        max_files=args.max_files,
        plot=not args.no_plot
    )