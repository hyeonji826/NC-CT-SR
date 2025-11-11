"""
NC-CE 페어링 검증
같은 환자의 같은 스캔인지 확인
"""
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

def check_pairing_quality(root):
    root = Path(root)
    pairs = pd.read_csv(root / "Data" / "pairs.csv")
    
    print("="*80)
    print("NC-CE 페어링 품질 검증")
    print("="*80)
    
    results = []
    
    for idx in range(min(10, len(pairs))):
        row = pairs.iloc[idx]
        case_id = str(row['id7']).strip().zfill(7)
        
        nc_path = root / row['input_nc_norm']
        ce_path = root / row['target_ce_norm']
        
        if not nc_path.exists() or not ce_path.exists():
            continue
        
        nc_img = sitk.ReadImage(str(nc_path))
        ce_img = sitk.ReadImage(str(ce_path))
        
        nc_arr = sitk.GetArrayFromImage(nc_img)
        ce_arr = sitk.GetArrayFromImage(ce_img)
        
        # 품질 지표
        nc_body = (nc_arr > 0.01).astype(float)
        ce_body = (ce_arr > 0.01).astype(float)
        
        nc_body_ratio = nc_body.mean()
        ce_body_ratio = ce_body.mean()
        
        # 중심 슬라이스 비교
        mid_nc = nc_arr.shape[0] // 2
        mid_ce = ce_arr.shape[0] // 2
        
        nc_slice = nc_arr[mid_nc]
        ce_slice = ce_arr[mid_ce]
        
        # Resize CE to NC size for comparison
        from skimage.transform import resize
        ce_resized = resize(ce_slice, nc_slice.shape, order=1, preserve_range=True, anti_aliasing=True)
        
        # Body overlap
        nc_body_slice = (nc_slice > 0.01).astype(float)
        ce_body_slice = (ce_resized > 0.01).astype(float)
        
        intersection = (nc_body_slice * ce_body_slice).sum()
        union = ((nc_body_slice + ce_body_slice) > 0).sum()
        iou = intersection / union if union > 0 else 0
        
        # Intensity correlation (body region only)
        nc_body_vals = nc_slice[nc_slice > 0.01]
        ce_body_vals = ce_resized[ce_resized > 0.01]
        
        if len(nc_body_vals) > 100 and len(ce_body_vals) > 100:
            # Sample same number
            n = min(len(nc_body_vals), len(ce_body_vals))
            nc_sample = np.random.choice(nc_body_vals, n, replace=False)
            ce_sample = np.random.choice(ce_body_vals, n, replace=False)
            correlation = np.corrcoef(nc_sample, ce_sample)[0, 1]
        else:
            correlation = 0
        
        results.append({
            'case_id': case_id,
            'nc_body_ratio': nc_body_ratio,
            'ce_body_ratio': ce_body_ratio,
            'body_iou': iou,
            'intensity_corr': correlation
        })
        
        print(f"\nCase {case_id}:")
        print(f"  NC body: {nc_body_ratio:.2%}, CE body: {ce_body_ratio:.2%}")
        print(f"  Body IoU: {iou:.3f} (1.0=완전일치, 0.0=겹침없음)")
        print(f"  Intensity correlation: {correlation:.3f} (1.0=완전상관, 0.0=무관)")
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(nc_slice, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'NC (mid slice)')
        axes[0].axis('off')
        
        axes[1].imshow(ce_resized, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'CE (mid slice, resized)')
        axes[1].axis('off')
        
        axes[2].imshow(np.abs(nc_slice - ce_resized), cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title(f'Difference (lower=better)')
        axes[2].axis('off')
        
        plt.suptitle(f'Case {case_id} - IoU: {iou:.3f}, Corr: {correlation:.3f}')
        plt.tight_layout()
        
        save_path = root / "Outputs" / "pairing_check" / f"{case_id}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    # Summary
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("페어링 품질 요약")
    print("="*80)
    print(f"\nBody IoU: {df['body_iou'].mean():.3f} ± {df['body_iou'].std():.3f}")
    print(f"Intensity correlation: {df['intensity_corr'].mean():.3f} ± {df['intensity_corr'].std():.3f}")
    
    print("\n진단:")
    if df['body_iou'].mean() < 0.3:
        print("❌ CRITICAL: Body 겹침이 거의 없습니다 (IoU < 0.3)")
        print("   → NC와 CE가 다른 환자이거나 완전히 다른 부위입니다")
        print("   → 이 데이터로는 학습 불가능")
    elif df['body_iou'].mean() < 0.5:
        print("⚠️  WARNING: Body 겹침이 적습니다 (IoU < 0.5)")
        print("   → NC와 CE가 다른 스캔 범위일 가능성")
    else:
        print("✓ Body 겹침 양호 (IoU >= 0.5)")
    
    if abs(df['intensity_corr'].mean()) < 0.3:
        print("❌ CRITICAL: 강도 상관관계가 거의 없습니다")
        print("   → NC와 CE가 완전히 다른 조직/환자")
    elif abs(df['intensity_corr'].mean()) < 0.5:
        print("⚠️  WARNING: 강도 상관관계가 약합니다")
    else:
        print("✓ 강도 상관관계 양호")
    
    print(f"\nVisualization saved: {root / 'Outputs' / 'pairing_check'}")
    
    return df

if __name__ == "__main__":
    check_pairing_quality(Path(r"E:\LD-CT SR"))