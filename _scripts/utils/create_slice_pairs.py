"""
물리적 Z축 좌표 기반 슬라이스 매칭
각 NC 슬라이스를 가장 가까운 CE 슬라이스와 페어링
"""
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

def find_matching_slices(nc_img, ce_img, max_distance_mm=5.0):
    """Find matching slices between NC and CE based on physical Z coordinates"""
    
    # Get Z coordinates for each slice
    nc_size = nc_img.GetSize()
    ce_size = ce_img.GetSize()
    
    nc_z_coords = []
    for i in range(nc_size[2]):
        z_coord = nc_img.TransformIndexToPhysicalPoint([0, 0, i])[2]
        nc_z_coords.append(z_coord)
    
    ce_z_coords = []
    for i in range(ce_size[2]):
        z_coord = ce_img.TransformIndexToPhysicalPoint([0, 0, i])[2]
        ce_z_coords.append(z_coord)
    
    nc_z_coords = np.array(nc_z_coords)
    ce_z_coords = np.array(ce_z_coords)
    
    # Find matches
    matches = []
    for nc_idx, nc_z in enumerate(nc_z_coords):
        # Find closest CE slice
        distances = np.abs(ce_z_coords - nc_z)
        ce_idx = int(np.argmin(distances))
        min_dist = distances[ce_idx]
        
        if min_dist <= max_distance_mm:
            matches.append({
                'nc_slice': nc_idx,
                'ce_slice': ce_idx,
                'distance_mm': float(min_dist),
                'z_coord': float(nc_z)
            })
    
    return matches

def create_slice_pairs(root, max_distance_mm=5.0):
    """Create slice-level pairs for all cases"""
    root = Path(root)
    pairs = pd.read_csv(root / "Data" / "pairs.csv")
    
    all_slice_pairs = []
    
    print(f"Creating slice-level pairs (max distance: {max_distance_mm}mm)...")
    
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Processing cases"):
        case_id = str(row['id7']).strip().zfill(7)
        
        nc_path = root / row['input_nc_norm']
        ce_path = root / row['target_ce_norm']
        
        if not nc_path.exists() or not ce_path.exists():
            continue
        
        try:
            nc_img = sitk.ReadImage(str(nc_path))
            ce_img = sitk.ReadImage(str(ce_path))
            
            matches = find_matching_slices(nc_img, ce_img, max_distance_mm)
            
            for match in matches:
                all_slice_pairs.append({
                    'case_id': case_id,
                    'nc_path': str(nc_path),
                    'ce_path': str(ce_path),
                    'nc_slice': match['nc_slice'],
                    'ce_slice': match['ce_slice'],
                    'distance_mm': match['distance_mm'],
                    'z_coord': match['z_coord']
                })
        
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            continue
    
    # Save
    df_slices = pd.DataFrame(all_slice_pairs)
    output_path = root / "Data" / "slice_pairs.csv"
    df_slices.to_csv(output_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"Slice-level pairs created!")
    print(f"{'='*80}")
    print(f"Total cases: {len(pairs)}")
    print(f"Total slice pairs: {len(df_slices)}")
    print(f"Average slices per case: {len(df_slices) / len(pairs):.1f}")
    print(f"Saved to: {output_path}")
    
    # Statistics
    print(f"\nDistance statistics:")
    print(f"  Mean: {df_slices['distance_mm'].mean():.2f}mm")
    print(f"  Max:  {df_slices['distance_mm'].max():.2f}mm")
    
    return df_slices

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--max-distance', type=float, default=5.0,
                       help='Maximum distance (mm) to consider slices as matching')
    args = parser.parse_args()
    
    create_slice_pairs(Path(args.root), args.max_distance)