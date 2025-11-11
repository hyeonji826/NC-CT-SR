"""
Super-Resolution 전처리: NC와 CE를 독립적으로 정규화
리샘플링 없이 각자의 해상도 유지
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# ==================== UTILS ====================
def ensure_dir(p): 
    p.mkdir(parents=True, exist_ok=True)

def clip_and_normalize(arr, wmin=-150, wmax=250, mode='zero_one'):
    """HU window clipping and normalization"""
    arr_clip = np.clip(arr, wmin, wmax)
    
    if mode == 'zero_one':
        arr_norm = (arr_clip - wmin) / (wmax - wmin)
        return np.clip(arr_norm, 0.0, 1.0)
    else:  # minus1_1
        arr_norm = 2.0 * (arr_clip - wmin) / (wmax - wmin) - 1.0
        return np.clip(arr_norm, -1.0, 1.0)

def voxel_radius_mm_to_pixels(spacing, radius_mm):
    """Convert mm radius to pixel radius"""
    return tuple(max(1, int(round(radius_mm / s))) for s in spacing)

def make_body_mask(img_hu_np, threshold=-200, spacing_xyz=None, closing_mm=3.0):
    """Create body mask with morphological operations"""
    mask = (img_hu_np > threshold).astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask)
    
    if spacing_xyz is not None:
        mask_img.SetSpacing(spacing_xyz[::-1])  # z,y,x -> x,y,z
        
        # Closing
        rad_pix = voxel_radius_mm_to_pixels(spacing_xyz, closing_mm)
        closing = sitk.BinaryMorphologicalClosingImageFilter()
        closing.SetKernelRadius(rad_pix[::-1])
        closing.SetForegroundValue(1)
        mask_img = closing.Execute(mask_img)
        
        # Fill holes
        fill = sitk.VotingBinaryHoleFillingImageFilter()
        fill.SetForegroundValue(1)
        fill.SetBackgroundValue(0)
        fill.SetRadius(rad_pix[::-1])
        fill.SetMajorityThreshold(1)
        mask_img = fill.Execute(mask_img)
    
    # Largest component
    cc = sitk.ConnectedComponent(mask_img)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(relabeled, 1, 1, 1, 0)
    
    return sitk.GetArrayFromImage(largest).astype(np.uint8)

def intensity_stats(arr, mask=None):
    """Compute intensity statistics"""
    if mask is not None:
        m = mask.astype(bool)
        vals = arr[m] if np.any(m) else arr
        body_ratio = float(np.mean(m))
    else:
        vals = arr.flatten()
        body_ratio = None
    
    if vals.size == 0:
        return {
            'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan,
            'p1': np.nan, 'p99': np.nan, 'body_ratio': body_ratio
        }
    
    return {
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'p1': float(np.percentile(vals, 1)),
        'p99': float(np.percentile(vals, 99)),
        'body_ratio': body_ratio
    }

# ==================== MAIN PROCESSING ====================
def process_case(case_id, root, args):
    """Process single case: normalize NC and CE independently"""
    root = Path(root)
    
    # Input paths
    nc_path = root / "Data" / "nii_cropped" / case_id / "NC_crop.nii"
    ce_path = root / "Data" / "nii_raw" / "CE_D" / case_id / "CE_raw.nii.gz"
    
    # Output paths
    nc_norm_out = root / "Data" / "nii_preproc_norm" / "NC" / case_id / "NC_norm.nii.gz"
    ce_norm_out = root / "Data" / "nii_preproc_norm" / "CE" / case_id / "CE_norm.nii.gz"
    
    result = {'id7': case_id}
    
    try:
        # Check if files exist
        if not nc_path.exists() or not ce_path.exists():
            result['status'] = 'missing_input'
            return result
        
        # === Process NC (Low-res input) ===
        nc_img = sitk.ReadImage(str(nc_path))
        nc_np = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        nc_spacing = tuple(nc_img.GetSpacing()[::-1])  # x,y,z -> z,y,x
        
        # Normalize NC
        nc_norm = clip_and_normalize(nc_np, args.window_min, args.window_max, args.norm_mode)
        
        # Make mask for NC
        nc_mask = None
        if args.masking:
            nc_mask = make_body_mask(nc_np, args.mask_threshold, nc_spacing, args.mask_closing_mm)
            # Apply mask (set background to 0)
            nc_norm[nc_mask == 0] = 0.0
        
        # Save NC normalized
        ensure_dir(nc_norm_out.parent)
        nc_norm_img = sitk.GetImageFromArray(nc_norm.astype(np.float32))
        nc_norm_img.CopyInformation(nc_img)
        sitk.WriteImage(nc_norm_img, str(nc_norm_out), useCompression=True)
        
        # === Process CE (High-res target) ===
        ce_img = sitk.ReadImage(str(ce_path))
        ce_np = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        ce_spacing = tuple(ce_img.GetSpacing()[::-1])  # x,y,z -> z,y,x
        
        # Normalize CE
        ce_norm = clip_and_normalize(ce_np, args.window_min, args.window_max, args.norm_mode)
        
        # Make mask for CE
        ce_mask = None
        if args.masking:
            ce_mask = make_body_mask(ce_np, args.mask_threshold, ce_spacing, args.mask_closing_mm)
            # Apply mask
            ce_norm[ce_mask == 0] = 0.0
        
        # Save CE normalized
        ensure_dir(ce_norm_out.parent)
        ce_norm_img = sitk.GetImageFromArray(ce_norm.astype(np.float32))
        ce_norm_img.CopyInformation(ce_img)
        sitk.WriteImage(ce_norm_img, str(ce_norm_out), useCompression=True)
        
        # === Compute statistics ===
        stats_nc = intensity_stats(nc_norm, nc_mask)
        stats_ce = intensity_stats(ce_norm, ce_mask)
        
        result.update({
            'status': 'success',
            'nc_norm': str(nc_norm_out),
            'ce_norm': str(ce_norm_out),
            'nc_shape': f"{nc_img.GetSize()}",
            'nc_spacing': f"({nc_spacing[0]:.3f}, {nc_spacing[1]:.3f}, {nc_spacing[2]:.3f})",
            'ce_shape': f"{ce_img.GetSize()}",
            'ce_spacing': f"({ce_spacing[0]:.3f}, {ce_spacing[1]:.3f}, {ce_spacing[2]:.3f})",
            'nc_min': stats_nc['min'], 'nc_max': stats_nc['max'],
            'nc_mean': stats_nc['mean'], 'nc_std': stats_nc['std'],
            'nc_p1': stats_nc['p1'], 'nc_p99': stats_nc['p99'],
            'nc_body_ratio': stats_nc['body_ratio'],
            'ce_min': stats_ce['min'], 'ce_max': stats_ce['max'],
            'ce_mean': stats_ce['mean'], 'ce_std': stats_ce['std'],
            'ce_p1': stats_ce['p1'], 'ce_p99': stats_ce['p99'],
            'ce_body_ratio': stats_ce['body_ratio'],
        })
        
        return result
        
    except Exception as e:
        result['status'] = f'error:{e}'
        return result

def main():
    parser = argparse.ArgumentParser(description='Super-Resolution Preprocessing')
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs', default='Data/pairs.csv')
    parser.add_argument('--window-min', type=float, default=-150)
    parser.add_argument('--window-max', type=float, default=250)
    parser.add_argument('--norm-mode', choices=['zero_one', 'minus1_1'], default='zero_one')
    parser.add_argument('--masking', dest='masking', action='store_true')
    parser.add_argument('--no-masking', dest='masking', action='store_false')
    parser.set_defaults(masking=True)
    parser.add_argument('--mask-threshold', type=float, default=-200)
    parser.add_argument('--mask-closing-mm', type=float, default=3.0)
    args = parser.parse_args()
    
    root = Path(args.root)
    pairs_path = root / args.pairs
    
    # Load pairs
    if not pairs_path.exists():
        print(f"Error: {pairs_path} not found")
        return
    
    df = pd.read_csv(pairs_path)
    key = "id7" if "id7" in df.columns else "id"
    case_ids = df[key].astype(str).str.strip().str.zfill(7).tolist()
    
    print(f"{'='*80}")
    print(f"Super-Resolution Preprocessing")
    print(f"{'='*80}")
    print(f"Total cases: {len(case_ids)}")
    print(f"Window: [{args.window_min}, {args.window_max}] HU")
    print(f"Normalization: {args.norm_mode}")
    print(f"Masking: {args.masking}")
    print(f"{'='*80}\n")
    
    # Process all cases
    results = []
    for case_id in tqdm(case_ids, desc="Processing cases", ncols=100):
        res = process_case(case_id, args.root, args)
        results.append(res)
    
    # Save results
    results_df = pd.DataFrame(results)
    log_path = root / "Outputs" / "reports" / "sr_preprocessing_log.csv"
    ensure_dir(log_path.parent)
    results_df.to_csv(log_path, index=False)
    
    # Save intensity statistics separately
    success_df = results_df[results_df['status'] == 'success'].copy()
    if len(success_df) > 0:
        intensity_cols = ['id7', 'nc_norm', 'ce_norm', 'nc_shape', 'nc_spacing', 
                         'ce_shape', 'ce_spacing',
                         'nc_min', 'nc_max', 'nc_mean', 'nc_std', 'nc_p1', 'nc_p99', 'nc_body_ratio',
                         'ce_min', 'ce_max', 'ce_mean', 'ce_std', 'ce_p1', 'ce_p99', 'ce_body_ratio']
        intensity_df = success_df[intensity_cols]
        intensity_path = root / "Outputs" / "reports" / "sr_intensity_summary.csv"
        intensity_df.to_csv(intensity_path, index=False)
    
    # Update pairs.csv with normalized paths
    if len(success_df) > 0:
        pairs_update = success_df[['id7', 'nc_norm', 'ce_norm']].copy()
        pairs_update.columns = ['id7', 'input_nc_norm', 'target_ce_norm']
        
        # Merge with original pairs
        df['id7'] = df[key].astype(str).str.strip().str.zfill(7)
        df_merged = df.merge(pairs_update, on='id7', how='left')
        
        # Update columns
        for col in ['input_nc_norm', 'target_ce_norm']:
            if col in df.columns:
                df_merged[col] = df_merged[col + '_y'].combine_first(df_merged[col + '_x'])
                df_merged = df_merged.drop(columns=[col + '_x', col + '_y'])
        
        df_merged.to_csv(pairs_path, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total processed: {len(results)}")
    print(f"\nStatus distribution:")
    print(results_df['status'].value_counts())
    
    if len(success_df) > 0:
        print(f"\n{'Statistics Summary':-^80}")
        print(f"NC body_ratio: mean={success_df['nc_body_ratio'].mean():.3f}, "
              f"min={success_df['nc_body_ratio'].min():.3f}, "
              f"max={success_df['nc_body_ratio'].max():.3f}")
        print(f"CE body_ratio: mean={success_df['ce_body_ratio'].mean():.3f}, "
              f"min={success_df['ce_body_ratio'].min():.3f}, "
              f"max={success_df['ce_body_ratio'].max():.3f}")
    
    print(f"\nLog saved: {log_path}")
    if len(success_df) > 0:
        print(f"Intensity stats: {intensity_path}")
        print(f"Pairs updated: {pairs_path}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()