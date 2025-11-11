"""
데이터셋 문제 진단 스크립트
PSNR이 낮은 원인 파악
"""
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

def check_data_quality(root):
    root = Path(root)
    pairs = pd.read_csv(root / "Data" / "pairs.csv")
    
    print("="*80)
    print("데이터 품질 체크")
    print("="*80)
    
    # 첫 3개 케이스 확인
    for idx in range(min(3, len(pairs))):
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
        
        print(f"\nCase {case_id}:")
        print(f"  NC shape: {nc_arr.shape}, range: [{nc_arr.min():.3f}, {nc_arr.max():.3f}], mean: {nc_arr.mean():.3f}")
        print(f"  CE shape: {ce_arr.shape}, range: [{ce_arr.min():.3f}, {ce_arr.max():.3f}], mean: {ce_arr.mean():.3f}")
        
        # 중간 슬라이스 시각화
        mid_nc = nc_arr.shape[0] // 2
        mid_ce = ce_arr.shape[0] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(nc_arr[mid_nc], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'NC - Slice {mid_nc}')
        axes[0].axis('off')
        
        axes[1].imshow(ce_arr[mid_ce], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'CE - Slice {mid_ce}')
        axes[1].axis('off')
        
        # Difference
        # CE에서 랜덤 패치 추출해서 비교
        if ce_arr.shape[1] >= 64 and ce_arr.shape[2] >= 64:
            h_start = (ce_arr.shape[1] - 64) // 2
            w_start = (ce_arr.shape[2] - 64) // 2
            ce_patch = ce_arr[mid_ce, h_start:h_start+64, w_start:w_start+64]
            
            if nc_arr.shape[1] >= 64 and nc_arr.shape[2] >= 64:
                nc_patch = nc_arr[mid_nc, h_start:h_start+64, w_start:w_start+64]
                diff = np.abs(ce_patch - nc_patch)
                
                axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.5)
                axes[2].set_title(f'Difference (mean: {diff.mean():.3f})')
                axes[2].axis('off')
        
        plt.suptitle(f'Case {case_id}')
        plt.tight_layout()
        
        save_path = root / "Outputs" / "debug" / f"case_{case_id}_check.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")
    
    # 전체 통계
    print("\n" + "="*80)
    print("전체 데이터셋 통계")
    print("="*80)
    
    all_nc_means = []
    all_ce_means = []
    all_nc_stds = []
    all_ce_stds = []
    
    for _, row in pairs.iterrows():
        nc_path = root / row['input_nc_norm']
        ce_path = root / row['target_ce_norm']
        
        if nc_path.exists() and ce_path.exists():
            nc_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(nc_path)))
            ce_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(ce_path)))
            
            all_nc_means.append(nc_arr.mean())
            all_ce_means.append(ce_arr.mean())
            all_nc_stds.append(nc_arr.std())
            all_ce_stds.append(ce_arr.std())
    
    print(f"\nNC - Mean: {np.mean(all_nc_means):.3f} ± {np.std(all_nc_means):.3f}")
    print(f"     Std:  {np.mean(all_nc_stds):.3f} ± {np.std(all_nc_stds):.3f}")
    print(f"\nCE - Mean: {np.mean(all_ce_means):.3f} ± {np.std(all_ce_means):.3f}")
    print(f"     Std:  {np.mean(all_ce_stds):.3f} ± {np.std(all_ce_stds):.3f}")
    
    # 진단
    print("\n" + "="*80)
    print("진단")
    print("="*80)
    
    nc_mean = np.mean(all_nc_means)
    ce_mean = np.mean(all_ce_means)
    
    if abs(nc_mean - ce_mean) < 0.05:
        print("⚠️  NC와 CE의 평균 강도가 거의 같습니다!")
        print("   → 조영 효과가 없거나 정규화에 문제가 있을 수 있습니다.")
    
    if nc_mean < 0.1 or ce_mean < 0.1:
        print("⚠️  강도 값이 너무 낮습니다 (거의 0에 가까움)")
        print("   → Body mask가 너무 aggressive하게 적용되었을 수 있습니다.")
    
    if nc_mean > 0.9 or ce_mean > 0.9:
        print("⚠️  강도 값이 너무 높습니다 (거의 1에 가까움)")
        print("   → 정규화 범위가 잘못되었을 수 있습니다.")
    
    print(f"\nDebug images saved in: {root / 'Outputs' / 'debug'}")

if __name__ == "__main__":
    check_data_quality(Path(r"E:\LD-CT SR"))