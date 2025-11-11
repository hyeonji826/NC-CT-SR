"""
Registration 실패 원인 진단
NC와 CE의 공간 정보, FOV, 내용 비교
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

def analyze_image(img, name):
    """이미지 공간 정보 및 내용 분석"""
    arr = sitk.GetArrayFromImage(img)
    
    info = {
        'name': name,
        'dim': f"{img.GetSize()}",
        'spacing_mm': f"({img.GetSpacing()[0]:.3f}, {img.GetSpacing()[1]:.3f}, {img.GetSpacing()[2]:.3f})",
        'origin_mm': f"({img.GetOrigin()[0]:.1f}, {img.GetOrigin()[1]:.1f}, {img.GetOrigin()[2]:.1f})",
    }
    
    # Intensity stats
    info['min'] = float(arr.min())
    info['max'] = float(arr.max())
    info['mean'] = float(arr.mean())
    info['p1'] = float(np.percentile(arr, 1))
    info['p99'] = float(np.percentile(arr, 99))
    
    # Body detection
    body_mask = arr > -200
    info['body_voxels'] = int(body_mask.sum())
    info['body_ratio'] = f"{body_mask.mean():.4f}"
    
    if body_mask.any():
        # Physical extent of body (bounding box)
        indices = np.argwhere(body_mask)
        z_min, y_min, x_min = indices.min(axis=0)
        z_max, y_max, x_max = indices.max(axis=0)
        
        # Convert index to physical coordinates
        corner_min = img.TransformIndexToPhysicalPoint([int(x_min), int(y_min), int(z_min)])
        corner_max = img.TransformIndexToPhysicalPoint([int(x_max), int(y_max), int(z_max)])
        
        info['body_bbox_min'] = f"({corner_min[0]:.1f}, {corner_min[1]:.1f}, {corner_min[2]:.1f})"
        info['body_bbox_max'] = f"({corner_max[0]:.1f}, {corner_max[1]:.1f}, {corner_max[2]:.1f})"
        
        # Center of body mass (physical coords)
        com_idx = indices.mean(axis=0)  # z, y, x
        com_phys = img.TransformContinuousIndexToPhysicalPoint([float(com_idx[2]), float(com_idx[1]), float(com_idx[0])])
        info['body_center'] = f"({com_phys[0]:.1f}, {com_phys[1]:.1f}, {com_phys[2]:.1f})"
    else:
        info['body_bbox_min'] = 'N/A'
        info['body_bbox_max'] = 'N/A'
        info['body_center'] = 'N/A'
    
    return info

def compare_pair(nc_path, ce_path, case_id):
    """NC-CE 쌍 비교"""
    print(f"\n{'='*80}")
    print(f"Case: {case_id}")
    print(f"{'='*80}")
    
    try:
        nc_img = sitk.ReadImage(str(nc_path))
        ce_img = sitk.ReadImage(str(ce_path))
        
        nc_info = analyze_image(nc_img, "NC")
        ce_info = analyze_image(ce_img, "CE")
        
        # Print side-by-side comparison
        print(f"\n{'Metric':<20} {'NC':<50} {'CE':<50}")
        print("-" * 120)
        
        for key in nc_info.keys():
            if key == 'name':
                continue
            print(f"{key:<20} {str(nc_info[key]):<50} {str(ce_info[key]):<50}")
        
        # Check overlap in physical space
        print(f"\n{'OVERLAP CHECK':-^120}")
        
        if nc_info['body_center'] != 'N/A' and ce_info['body_center'] != 'N/A':
            # Parse centers
            nc_center = [float(x) for x in nc_info['body_center'].strip('()').split(',')]
            ce_center = [float(x) for x in ce_info['body_center'].strip('()').split(',')]
            
            distance = np.sqrt(sum((a - b)**2 for a, b in zip(nc_center, ce_center)))
            print(f"Distance between body centers: {distance:.1f} mm")
            
            if distance > 500:
                print("⚠️  WARNING: Centers are >500mm apart - likely NO overlap!")
            elif distance > 200:
                print("⚠️  CAUTION: Centers are >200mm apart - limited overlap possible")
            else:
                print("✓ Centers are close - good overlap expected")
        else:
            print("❌ Cannot compute overlap - one or both images have no body content!")
        
        # Check if NC crop is reasonable
        nc_arr = sitk.GetArrayFromImage(nc_img)
        ce_arr = sitk.GetArrayFromImage(ce_img)
        
        print(f"\n{'CONTENT CHECK':-^120}")
        print(f"NC total voxels: {nc_arr.size:,}")
        print(f"CE total voxels: {ce_arr.size:,}")
        print(f"NC body ratio: {nc_info['body_ratio']}")
        print(f"CE body ratio: {ce_info['body_ratio']}")
        
        if float(nc_info['body_ratio']) < 0.01:
            print("❌ NC has almost no body content - crop may have failed!")
        elif float(nc_info['body_ratio']) < 0.1:
            print("⚠️  NC has very little body content - check crop parameters")
        else:
            print("✓ NC has reasonable body content")
            
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    # Read failed cases from resample log
    log_path = Path(r"E:\LD-CT SR\Outputs\reports\resample_log.csv")
    
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        return
    
    df = pd.read_csv(log_path)
    
    # Filter empty cases
    empty_cases = df[df['status'] == 'empty_after_resample'].copy()
    print(f"Found {len(empty_cases)} empty cases to diagnose")
    
    # Analyze first 3 cases in detail
    num_to_check = min(3, len(empty_cases))
    print(f"\nAnalyzing first {num_to_check} cases in detail...")
    
    for idx, row in empty_cases.head(num_to_check).iterrows():
        nc_path = Path(row['nc_path'])
        ce_path = Path(row['ce_path'])
        case_id = row['id7']
        
        compare_pair(nc_path, ce_path, case_id)
    
    # Summary statistics for all failed cases
    print(f"\n\n{'SUMMARY STATISTICS FOR ALL FAILED CASES':#^120}")
    print(f"\nTotal failed: {len(empty_cases)}")
    
    # Try to gather body_ratio from original files
    print("\nChecking original NC crop files...")
    body_ratios = []
    
    for idx, row in empty_cases.head(10).iterrows():  # Check first 10
        try:
            nc_img = sitk.ReadImage(str(row['nc_path']))
            arr = sitk.GetArrayFromImage(nc_img)
            body_ratio = float((arr > -200).mean())
            body_ratios.append(body_ratio)
        except:
            continue
    
    if body_ratios:
        print(f"Original NC body_ratio stats (n={len(body_ratios)}):")
        print(f"  Mean: {np.mean(body_ratios):.4f}")
        print(f"  Min:  {np.min(body_ratios):.4f}")
        print(f"  Max:  {np.max(body_ratios):.4f}")
        
        if np.mean(body_ratios) < 0.01:
            print("\n❌ CRITICAL: Original NC crops are EMPTY!")
            print("   → The problem is in the cropping step (step 02), NOT resampling")
        elif np.mean(body_ratios) > 0.1:
            print("\n⚠️  Original NC crops look OK, but resampling failed")
            print("   → The problem is in spatial alignment")
    
    print("\n" + "="*120)
    print("DIAGNOSIS COMPLETE")
    print("="*120)

if __name__ == "__main__":
    main()