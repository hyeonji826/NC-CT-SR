"""
NC와 CE의 해상도 차이 확인
"""
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk

def analyze_resolution_ratio(root):
    root = Path(root)
    pairs_path = root / "Data" / "pairs.csv"
    
    if not pairs_path.exists():
        print(f"Error: {pairs_path} not found")
        return
    
    df = pd.read_csv(pairs_path)
    key = "id7" if "id7" in df.columns else "id"
    
    results = []
    
    print("Analyzing first 5 cases...\n")
    
    for idx, row in df.head(5).iterrows():
        case_id = str(row[key]).strip().zfill(7)
        
        nc_path = root / "Data" / "nii_cropped" / case_id / "NC_crop.nii"
        ce_path = root / "Data" / "nii_raw" / "CE_D" / case_id / "CE_raw.nii.gz"
        
        if not nc_path.exists() or not ce_path.exists():
            continue
        
        try:
            nc_img = sitk.ReadImage(str(nc_path))
            ce_img = sitk.ReadImage(str(ce_path))
            
            nc_size = nc_img.GetSize()
            ce_size = ce_img.GetSize()
            nc_spacing = nc_img.GetSpacing()
            ce_spacing = ce_img.GetSpacing()
            
            # Calculate ratios
            ratio_x = nc_spacing[0] / ce_spacing[0]
            ratio_y = nc_spacing[1] / ce_spacing[1]
            ratio_z = nc_spacing[2] / ce_spacing[2]
            
            print(f"Case: {case_id}")
            print(f"  NC: size={nc_size}, spacing=({nc_spacing[0]:.3f}, {nc_spacing[1]:.3f}, {nc_spacing[2]:.3f})")
            print(f"  CE: size={ce_size}, spacing=({ce_spacing[0]:.3f}, {ce_spacing[1]:.3f}, {ce_spacing[2]:.3f})")
            print(f"  Ratio (NC/CE): x={ratio_x:.2f}, y={ratio_y:.2f}, z={ratio_z:.2f}")
            print(f"  → {'NC is LOWER resolution' if ratio_x > 1 else 'CE is LOWER resolution'}\n")
            
            results.append({
                'id7': case_id,
                'nc_spacing_x': nc_spacing[0],
                'nc_spacing_y': nc_spacing[1],
                'nc_spacing_z': nc_spacing[2],
                'ce_spacing_x': ce_spacing[0],
                'ce_spacing_y': ce_spacing[1],
                'ce_spacing_z': ce_spacing[2],
                'ratio_x': ratio_x,
                'ratio_y': ratio_y,
                'ratio_z': ratio_z,
            })
            
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
            continue
    
    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"\nAverage spacing ratios (NC/CE):")
        print(f"  X-axis: {df_res['ratio_x'].mean():.2f} ± {df_res['ratio_x'].std():.2f}")
        print(f"  Y-axis: {df_res['ratio_y'].mean():.2f} ± {df_res['ratio_y'].std():.2f}")
        print(f"  Z-axis: {df_res['ratio_z'].mean():.2f} ± {df_res['ratio_z'].std():.2f}")
        
        avg_ratio_xy = df_res[['ratio_x', 'ratio_y']].mean().mean()
        print(f"\n⭐ Average in-plane ratio: {avg_ratio_xy:.2f}x")
        
        if 1.5 <= avg_ratio_xy < 2.5:
            print("\n✅ Recommendation: Use SwinIR x2 model")
            print("   Download: 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth")
        elif 2.5 <= avg_ratio_xy < 5.0:
            print("\n✅ Recommendation: Use SwinIR x4 model")
            print("   Download: 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth")
        elif avg_ratio_xy >= 5.0:
            print("\n⚠️  Warning: Very large resolution difference (>5x)")
            print("   Consider: 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth + multi-stage SR")
        elif avg_ratio_xy < 1.5:
            print("\n⚠️  Warning: Resolution difference is small (<1.5x)")
            print("   NC might already be high-res, or CE is lower-res than expected")
        
        # Check if current weights are appropriate
        print("\n" + "="*80)
        print("CURRENT WEIGHTS ANALYSIS")
        print("="*80)
        print("Your downloaded weights: 004_grayDN_DFWB_s128w8_SwinIR-M_noise*.pth")
        print("  → These are for DENOISING, not super-resolution")
        print("  → ❌ NOT suitable for your SR task")
        print("\nYou need to download SR weights instead:")
        print("  → 001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth  (for ~2x upsampling)")
        print("  → 001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth  (for ~4x upsampling)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    args = parser.parse_args()
    
    analyze_resolution_ratio(args.root)