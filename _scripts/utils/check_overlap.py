"""
NC-CE 쌍의 Z축 overlap 확인 및 필터링
겹치는 구간이 있는 케이스만 선별
"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

def get_z_range(img, threshold=-200):
    """Body region의 Z축 물리적 범위 계산"""
    arr = sitk.GetArrayFromImage(img)
    body_mask = arr > threshold
    
    if not body_mask.any():
        return None, None
    
    # Z축 인덱스 범위
    z_indices = np.where(body_mask.any(axis=(1, 2)))[0]
    z_min_idx = int(z_indices.min())
    z_max_idx = int(z_indices.max())
    
    # 물리적 좌표로 변환
    # SimpleITK: (x, y, z), numpy: (z, y, x)
    z_min_phys = img.TransformIndexToPhysicalPoint([0, 0, z_min_idx])[2]
    z_max_phys = img.TransformIndexToPhysicalPoint([0, 0, z_max_idx])[2]
    
    # Ensure min < max
    if z_min_phys > z_max_phys:
        z_min_phys, z_max_phys = z_max_phys, z_min_phys
    
    return z_min_phys, z_max_phys

def compute_overlap(nc_range, ce_range):
    """두 범위의 겹침 계산"""
    if None in nc_range or None in ce_range:
        return 0.0, 0.0, 0.0
    
    nc_min, nc_max = nc_range
    ce_min, ce_max = ce_range
    
    # Overlap range
    overlap_min = max(nc_min, ce_min)
    overlap_max = min(nc_max, ce_max)
    
    if overlap_max <= overlap_min:
        # No overlap
        return 0.0, 0.0, abs(overlap_min - overlap_max)
    
    # Overlap length
    overlap_length = overlap_max - overlap_min
    nc_length = nc_max - nc_min
    ce_length = ce_max - ce_min
    
    # Overlap ratio
    overlap_ratio_nc = overlap_length / nc_length if nc_length > 0 else 0
    overlap_ratio_ce = overlap_length / ce_length if ce_length > 0 else 0
    
    return overlap_length, min(overlap_ratio_nc, overlap_ratio_ce), 0.0

def check_all_pairs(root, pairs_csv):
    """모든 NC-CE 쌍의 overlap 확인"""
    root = Path(root)
    pairs_path = root / pairs_csv
    
    # Load pairs
    if not pairs_path.exists():
        print(f"Error: {pairs_path} not found")
        return
    
    df = pd.read_csv(pairs_path)
    key = "id7" if "id7" in df.columns else "id"
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking overlap"):
        case_id = str(row[key]).strip().zfill(7)
        
        # Build paths
        nc_path = root / "Data" / "nii_cropped" / case_id / "NC_crop.nii"
        ce_path = root / "Data" / "nii_raw" / "CE_D" / case_id / "CE_raw.nii.gz"
        
        if not nc_path.exists() or not ce_path.exists():
            results.append({
                'id7': case_id,
                'status': 'missing_files',
                'nc_z_min': None, 'nc_z_max': None,
                'ce_z_min': None, 'ce_z_max': None,
                'overlap_mm': None,
                'overlap_ratio': None,
                'gap_mm': None
            })
            continue
        
        try:
            # Read images
            nc_img = sitk.ReadImage(str(nc_path))
            ce_img = sitk.ReadImage(str(ce_path))
            
            # Get Z ranges
            nc_z_range = get_z_range(nc_img)
            ce_z_range = get_z_range(ce_img)
            
            # Compute overlap
            overlap_mm, overlap_ratio, gap_mm = compute_overlap(nc_z_range, ce_z_range)
            
            # Status
            if overlap_mm > 50:  # >50mm overlap
                status = 'good'
            elif overlap_mm > 0:
                status = 'marginal'
            elif gap_mm > 100:
                status = 'large_gap'
            else:
                status = 'no_overlap'
            
            results.append({
                'id7': case_id,
                'status': status,
                'nc_z_min': f"{nc_z_range[0]:.1f}" if nc_z_range[0] else None,
                'nc_z_max': f"{nc_z_range[1]:.1f}" if nc_z_range[1] else None,
                'ce_z_min': f"{ce_z_range[0]:.1f}" if ce_z_range[0] else None,
                'ce_z_max': f"{ce_z_range[1]:.1f}" if ce_z_range[1] else None,
                'overlap_mm': f"{overlap_mm:.1f}",
                'overlap_ratio': f"{overlap_ratio:.3f}",
                'gap_mm': f"{gap_mm:.1f}" if gap_mm > 0 else "0.0"
            })
            
        except Exception as e:
            results.append({
                'id7': case_id,
                'status': f'error:{e}',
                'nc_z_min': None, 'nc_z_max': None,
                'ce_z_min': None, 'ce_z_max': None,
                'overlap_mm': None,
                'overlap_ratio': None,
                'gap_mm': None
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = root / "Outputs" / "reports" / "z_overlap_check.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("Z-AXIS OVERLAP SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal cases: {len(results_df)}")
    print(f"\nStatus distribution:")
    print(results_df['status'].value_counts())
    
    # Statistics for numeric columns
    results_df['overlap_mm_num'] = pd.to_numeric(results_df['overlap_mm'], errors='coerce')
    results_df['overlap_ratio_num'] = pd.to_numeric(results_df['overlap_ratio'], errors='coerce')
    results_df['gap_mm_num'] = pd.to_numeric(results_df['gap_mm'], errors='coerce')
    
    print(f"\nOverlap statistics (mm):")
    print(results_df['overlap_mm_num'].describe())
    
    print(f"\nOverlap ratio statistics:")
    print(results_df['overlap_ratio_num'].describe())
    
    # Filter good cases
    good_cases = results_df[results_df['status'].isin(['good', 'marginal'])]
    print(f"\n{'='*80}")
    print(f"USABLE CASES: {len(good_cases)} / {len(results_df)} ({100*len(good_cases)/len(results_df):.1f}%)")
    print(f"{'='*80}")
    
    # Save filtered pairs
    if len(good_cases) > 0:
        filtered_pairs_path = root / "Data" / "pairs_filtered.csv"
        good_cases[['id7']].to_csv(filtered_pairs_path, index=False)
        print(f"\nFiltered pairs saved to: {filtered_pairs_path}")
    
    print(f"\nFull results saved to: {output_path}")
    
    return results_df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check Z-axis overlap between NC and CE")
    parser.add_argument("--root", default=r"E:\LD-CT SR")
    parser.add_argument("--pairs", default="Data/pairs.csv")
    args = parser.parse_args()
    
    check_all_pairs(args.root, args.pairs)

if __name__ == "__main__":
    main()