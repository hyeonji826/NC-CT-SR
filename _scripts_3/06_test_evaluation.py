"""
06_test_evaluation.py
Test 데이터셋 평가

목표:
1. Test set에서 정량적 평가 (PSNR, SSIM, MSE)
2. 정성적 평가 (시각화)
3. 장기별 분석 (Aorta, Liver bright, Liver dark)
4. CE Ground Truth와 비교
5. 통계 분석 및 리포트 생성

워크플로우:
- NC → Enhanced NC (모델)
- Enhanced NC vs CE (Ground Truth) 비교
- 장기별 성능 측정
"""

import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import mean_squared_error as mse_metric
import pandas as pd
import json
import sys
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from models import StructurePreservingStyleTransfer


# ============================================================
# Model Wrapper
# ============================================================

class TestModel:
    """평가용 모델 래퍼"""
    
    def __init__(self, checkpoint_path, device='cuda', base_channels=64):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = StructurePreservingStyleTransfer(base_channels=base_channels)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 모델 로드: {Path(checkpoint_path).name}")
        print(f"   Device: {self.device}")
    
    @torch.no_grad()
    def enhance_volume(self, nc_image, ce_reference, style_alpha=1.0):
        """
        3D 볼륨 전체 향상
        
        Args:
            nc_image: NC SimpleITK Image
            ce_reference: CE SimpleITK Image (style reference)
            style_alpha: 스타일 강도
        
        Returns:
            enhanced_image: SimpleITK Image
        """
        nc_arr = sitk.GetArrayFromImage(nc_image)
        ce_arr = sitk.GetArrayFromImage(ce_reference)
        
        num_slices = nc_arr.shape[0]
        enhanced_slices = []
        
        for i in range(num_slices):
            # To tensor [1, 1, H, W]
            nc_slice = torch.from_numpy(nc_arr[i]).float().unsqueeze(0).unsqueeze(0).to(self.device)
            ce_slice = torch.from_numpy(ce_arr[i]).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # 모델 추론
            enhanced_slice = self.model(nc_slice, ce_slice, alpha=style_alpha)
            enhanced_slice = torch.clamp(enhanced_slice, 0, 1)
            
            enhanced_slices.append(enhanced_slice.cpu().numpy()[0, 0])
        
        # Stack
        enhanced_arr = np.stack(enhanced_slices, axis=0)
        
        # To SimpleITK
        enhanced_image = sitk.GetImageFromArray(enhanced_arr)
        enhanced_image.CopyInformation(nc_image)
        
        return enhanced_image


# ============================================================
# Metrics
# ============================================================

def compute_metrics(pred, target, data_range=1.0):
    """
    PSNR, SSIM, MSE 계산
    
    Args:
        pred: 예측 이미지 (numpy array)
        target: 타겟 이미지 (numpy array)
        data_range: 데이터 범위
    
    Returns:
        dict: 메트릭 딕셔너리
    """
    psnr = psnr_metric(target, pred, data_range=data_range)
    ssim = ssim_metric(target, pred, data_range=data_range)
    mse = mse_metric(target, pred)
    mae = np.mean(np.abs(target - pred))
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse,
        'mae': mae
    }


def compute_organ_metrics(pred, target, seg_mask, organ_value):
    """
    특정 장기 영역에서만 메트릭 계산
    
    Args:
        pred: 예측 이미지
        target: 타겟 이미지
        seg_mask: 세그멘테이션 마스크
        organ_value: 장기 라벨 값 (e.g., 1=liver, 2=aorta)
    
    Returns:
        dict or None
    """
    mask = (seg_mask == organ_value)
    
    if mask.sum() == 0:
        return None
    
    pred_organ = pred[mask]
    target_organ = target[mask]
    
    # 평균 intensity difference
    mean_diff = np.mean(target_organ - pred_organ)
    abs_diff = np.mean(np.abs(target_organ - pred_organ))
    
    # Contrast (표준편차)
    pred_contrast = np.std(pred_organ)
    target_contrast = np.std(target_organ)
    
    return {
        'mean_diff': mean_diff,
        'abs_diff': abs_diff,
        'pred_contrast': pred_contrast,
        'target_contrast': target_contrast,
        'pixel_count': mask.sum()
    }


# ============================================================
# Test Patient Processing
# ============================================================

def test_patient(patient_id, nc_path, ce_path, aorta_seg_path, liver_seg_path,
                weight_map_path, model, output_dir, save_nifti=True):
    """
    Test 환자 1명 평가
    
    Returns:
        results: dict
    """
    # Load images
    nc_image = sitk.ReadImage(str(nc_path))
    ce_image = sitk.ReadImage(str(ce_path))
    
    nc_arr = sitk.GetArrayFromImage(nc_image)
    ce_arr = sitk.GetArrayFromImage(ce_image)
    
    # Load segmentations
    aorta_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(aorta_seg_path)))
    liver_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(liver_seg_path)))
    weight_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(weight_map_path)))
    
    # 모델 추론
    enhanced_image = model.enhance_volume(nc_image, ce_image, style_alpha=1.0)
    enhanced_arr = sitk.GetArrayFromImage(enhanced_image)
    
    # Save enhanced
    if save_nifti:
        patient_output_dir = output_dir / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        enhanced_path = patient_output_dir / 'Enhanced_NC.nii.gz'
        sitk.WriteImage(enhanced_image, str(enhanced_path))
    
    # ========================================
    # 1. Overall Metrics (전체 볼륨)
    # ========================================
    
    overall_metrics = compute_metrics(enhanced_arr, ce_arr)
    
    # ========================================
    # 2. Slice-wise Metrics
    # ========================================
    
    slice_metrics = []
    for i in range(nc_arr.shape[0]):
        metrics = compute_metrics(enhanced_arr[i], ce_arr[i])
        metrics['slice_idx'] = i
        slice_metrics.append(metrics)
    
    # ========================================
    # 3. Organ-specific Metrics
    # ========================================
    
    # Aorta
    aorta_metrics = compute_organ_metrics(
        enhanced_arr, ce_arr, aorta_arr, organ_value=1
    )
    
    # Liver (전체)
    liver_metrics = compute_organ_metrics(
        enhanced_arr, ce_arr, liver_arr, organ_value=1
    )
    
    # Liver bright (weight > 0.6)
    liver_bright_mask = (liver_arr == 1) & (weight_arr > 0.6)
    if liver_bright_mask.sum() > 0:
        liver_bright_metrics = {
            'mean_diff': np.mean(ce_arr[liver_bright_mask] - enhanced_arr[liver_bright_mask]),
            'abs_diff': np.mean(np.abs(ce_arr[liver_bright_mask] - enhanced_arr[liver_bright_mask])),
            'pixel_count': liver_bright_mask.sum()
        }
    else:
        liver_bright_metrics = None
    
    # Liver dark (weight < 0.4, tumor region)
    liver_dark_mask = (liver_arr == 1) & (weight_arr < 0.4)
    if liver_dark_mask.sum() > 0:
        liver_dark_metrics = {
            'mean_diff': np.mean(ce_arr[liver_dark_mask] - enhanced_arr[liver_dark_mask]),
            'abs_diff': np.mean(np.abs(ce_arr[liver_dark_mask] - enhanced_arr[liver_dark_mask])),
            'pixel_count': liver_dark_mask.sum()
        }
    else:
        liver_dark_metrics = None
    
    # ========================================
    # 4. Contrast Analysis (대비 증가 확인)
    # ========================================
    
    # Liver bright vs dark 대비
    if liver_bright_mask.sum() > 0 and liver_dark_mask.sum() > 0:
        # NC
        nc_bright_mean = nc_arr[liver_bright_mask].mean()
        nc_dark_mean = nc_arr[liver_dark_mask].mean()
        nc_contrast = nc_bright_mean - nc_dark_mean
        
        # Enhanced
        enh_bright_mean = enhanced_arr[liver_bright_mask].mean()
        enh_dark_mean = enhanced_arr[liver_dark_mask].mean()
        enh_contrast = enh_bright_mean - enh_dark_mean
        
        # CE (Ground Truth)
        ce_bright_mean = ce_arr[liver_bright_mask].mean()
        ce_dark_mean = ce_arr[liver_dark_mask].mean()
        ce_contrast = ce_bright_mean - ce_dark_mean
        
        contrast_analysis = {
            'nc_contrast': nc_contrast,
            'enhanced_contrast': enh_contrast,
            'ce_contrast': ce_contrast,
            'contrast_improvement': enh_contrast - nc_contrast,
            'contrast_to_ce_ratio': enh_contrast / ce_contrast if ce_contrast > 0 else 0
        }
    else:
        contrast_analysis = None
    
    # ========================================
    # Results
    # ========================================
    
    results = {
        'patient_id': patient_id,
        'overall': overall_metrics,
        'slice_wise': slice_metrics,
        'organs': {
            'aorta': aorta_metrics,
            'liver': liver_metrics,
            'liver_bright': liver_bright_metrics,
            'liver_dark': liver_dark_metrics
        },
        'contrast': contrast_analysis
    }
    
    return results, enhanced_arr


# ============================================================
# Visualization
# ============================================================

def visualize_test_result(patient_id, nc_arr, ce_arr, enhanced_arr,
                         aorta_arr, liver_arr, weight_arr, results,
                         output_path, num_samples=6):
    """
    Test 결과 시각화 (상세)
    """
    num_slices = nc_arr.shape[0]
    
    # Select slices with max organ coverage
    valid_slices = []
    for i in range(num_slices):
        organ_pixels = aorta_arr[i].sum() + liver_arr[i].sum()
        if organ_pixels > 100:
            valid_slices.append((i, organ_pixels))
    
    valid_slices.sort(key=lambda x: x[1], reverse=True)
    selected_slices = [s[0] for s in valid_slices[:num_samples]]
    
    if len(selected_slices) < num_samples:
        selected_slices = np.linspace(
            int(num_slices * 0.3),
            int(num_slices * 0.7),
            num_samples,
            dtype=int
        ).tolist()
    
    # Plot
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(6, num_samples, hspace=0.3, wspace=0.2)
    
    # Title
    overall = results['overall']
    contrast = results['contrast']
    
    title = f'Patient {patient_id} - Test Evaluation\n'
    title += f'Overall: PSNR={overall["psnr"]:.2f} dB | SSIM={overall["ssim"]:.4f} | MSE={overall["mse"]:.6f}\n'
    
    if contrast:
        title += f'Contrast: NC={contrast["nc_contrast"]:.3f} → Enhanced={contrast["enhanced_contrast"]:.3f} '
        title += f'(CE={contrast["ce_contrast"]:.3f}) | Improvement={contrast["contrast_improvement"]:.3f}'
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    for col, slice_idx in enumerate(selected_slices):
        slice_metrics = results['slice_wise'][slice_idx]
        
        # Row 1: NC
        ax1 = fig.add_subplot(gs[0, col])
        ax1.imshow(nc_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        ax1.set_title(f'Slice {slice_idx}\nNC (Original)', fontsize=10, fontweight='bold', color='blue')
        ax1.axis('off')
        
        # Row 2: CE (Ground Truth)
        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(ce_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        ax2.set_title('CE (Ground Truth)', fontsize=10, fontweight='bold', color='red')
        ax2.axis('off')
        
        # Row 3: Enhanced
        ax3 = fig.add_subplot(gs[2, col])
        ax3.imshow(enhanced_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        ax3.set_title(f'Enhanced NC\nPSNR: {slice_metrics["psnr"]:.2f} dB',
                     fontsize=10, fontweight='bold', color='green')
        ax3.axis('off')
        
        # Row 4: Difference (Enhanced vs CE)
        ax4 = fig.add_subplot(gs[3, col])
        diff = np.abs(enhanced_arr[slice_idx] - ce_arr[slice_idx])
        im = ax4.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        ax4.set_title(f'|Enhanced - CE|\nSSIM: {slice_metrics["ssim"]:.4f}', fontsize=10)
        ax4.axis('off')
        
        # Row 5: Weight Map
        ax5 = fig.add_subplot(gs[4, col])
        im_weight = ax5.imshow(weight_arr[slice_idx], cmap='jet', vmin=0.1, vmax=1.0)
        ax5.set_title('Weight Map', fontsize=10, color='purple')
        ax5.axis('off')
        
        # Row 6: Segmentation overlay
        ax6 = fig.add_subplot(gs[5, col])
        ax6.imshow(enhanced_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        
        # Overlay
        aorta_mask = np.ma.masked_where(aorta_arr[slice_idx] == 0, aorta_arr[slice_idx])
        liver_mask = np.ma.masked_where(liver_arr[slice_idx] == 0, liver_arr[slice_idx])
        
        ax6.imshow(aorta_mask, cmap='Reds', alpha=0.4)
        ax6.imshow(liver_mask, cmap='Greens', alpha=0.4)
        
        aorta_px = int(aorta_arr[slice_idx].sum())
        liver_px = int(liver_arr[slice_idx].sum())
        ax6.set_title(f'Segmentation\nAorta: {aorta_px} | Liver: {liver_px}', fontsize=10)
        ax6.axis('off')
    
    # Colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.35, 0.01, 0.15])
    fig.colorbar(im, cax=cbar_ax1, label='Difference')
    
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.01, 0.15])
    fig.colorbar(im_weight, cax=cbar_ax2, label='Weight')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Statistics and Report
# ============================================================

def generate_statistics(all_results, output_dir):
    """
    전체 통계 및 리포트 생성
    """
    print("\n" + "="*80)
    print("통계 분석 및 리포트 생성")
    print("="*80)
    
    # ========================================
    # 1. Overall Statistics
    # ========================================
    
    overall_stats = {
        'psnr': [],
        'ssim': [],
        'mse': [],
        'mae': []
    }
    
    for result in all_results:
        overall = result['overall']
        overall_stats['psnr'].append(overall['psnr'])
        overall_stats['ssim'].append(overall['ssim'])
        overall_stats['mse'].append(overall['mse'])
        overall_stats['mae'].append(overall['mae'])
    
    stats_summary = {}
    for key in overall_stats:
        values = overall_stats[key]
        stats_summary[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Print
    print("\n📊 Overall Statistics:")
    print(f"  PSNR: {stats_summary['psnr']['mean']:.2f} ± {stats_summary['psnr']['std']:.2f} dB")
    print(f"  SSIM: {stats_summary['ssim']['mean']:.4f} ± {stats_summary['ssim']['std']:.4f}")
    print(f"  MSE:  {stats_summary['mse']['mean']:.6f} ± {stats_summary['mse']['std']:.6f}")
    print(f"  MAE:  {stats_summary['mae']['mean']:.6f} ± {stats_summary['mae']['std']:.6f}")
    
    # ========================================
    # 2. Organ-specific Statistics
    # ========================================
    
    organ_stats = {
        'aorta': {'abs_diff': [], 'pixel_count': []},
        'liver': {'abs_diff': [], 'pixel_count': []},
        'liver_bright': {'abs_diff': [], 'pixel_count': []},
        'liver_dark': {'abs_diff': [], 'pixel_count': []}
    }
    
    for result in all_results:
        organs = result['organs']
        
        for organ_name in organ_stats.keys():
            organ_data = organs.get(organ_name)
            if organ_data and 'abs_diff' in organ_data:
                organ_stats[organ_name]['abs_diff'].append(organ_data['abs_diff'])
                organ_stats[organ_name]['pixel_count'].append(organ_data['pixel_count'])
    
    print("\n🎯 Organ-specific Statistics:")
    for organ_name, data in organ_stats.items():
        if len(data['abs_diff']) > 0:
            mean_diff = np.mean(data['abs_diff'])
            std_diff = np.std(data['abs_diff'])
            print(f"  {organ_name.capitalize():15s}: MAE = {mean_diff:.4f} ± {std_diff:.4f}")
    
    # ========================================
    # 3. Contrast Analysis
    # ========================================
    
    contrast_stats = {
        'nc_contrast': [],
        'enhanced_contrast': [],
        'ce_contrast': [],
        'improvement': []
    }
    
    for result in all_results:
        contrast = result.get('contrast')
        if contrast:
            contrast_stats['nc_contrast'].append(contrast['nc_contrast'])
            contrast_stats['enhanced_contrast'].append(contrast['enhanced_contrast'])
            contrast_stats['ce_contrast'].append(contrast['ce_contrast'])
            contrast_stats['improvement'].append(contrast['contrast_improvement'])
    
    if len(contrast_stats['improvement']) > 0:
        print("\n🔍 Contrast Analysis (Liver Bright vs Dark):")
        print(f"  NC Contrast:       {np.mean(contrast_stats['nc_contrast']):.4f} ± {np.std(contrast_stats['nc_contrast']):.4f}")
        print(f"  Enhanced Contrast: {np.mean(contrast_stats['enhanced_contrast']):.4f} ± {np.std(contrast_stats['enhanced_contrast']):.4f}")
        print(f"  CE Contrast (GT):  {np.mean(contrast_stats['ce_contrast']):.4f} ± {np.std(contrast_stats['ce_contrast']):.4f}")
        print(f"  Improvement:       {np.mean(contrast_stats['improvement']):.4f} ± {np.std(contrast_stats['improvement']):.4f}")
    
    # ========================================
    # 4. Save to JSON
    # ========================================
    
    summary = {
        'overall_statistics': stats_summary,
        'organ_statistics': organ_stats,
        'contrast_statistics': contrast_stats,
        'num_patients': len(all_results)
    }
    
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 통계 저장: {output_dir / 'statistics.json'}")
    
    # ========================================
    # 5. Save to CSV
    # ========================================
    
    # Patient-wise results
    csv_data = []
    for result in all_results:
        row = {
            'patient_id': result['patient_id'],
            'psnr': result['overall']['psnr'],
            'ssim': result['overall']['ssim'],
            'mse': result['overall']['mse'],
            'mae': result['overall']['mae']
        }
        
        # Organ metrics
        for organ_name, organ_data in result['organs'].items():
            if organ_data:
                row[f'{organ_name}_mae'] = organ_data.get('abs_diff', np.nan)
        
        # Contrast
        contrast = result.get('contrast')
        if contrast:
            row['contrast_improvement'] = contrast['contrast_improvement']
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_dir / 'results.csv', index=False)
    
    print(f"💾 결과 CSV 저장: {output_dir / 'results.csv'}")
    
    # ========================================
    # 6. Visualization: Box plots
    # ========================================
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Test Set Statistics', fontsize=16, fontweight='bold')
    
    # PSNR
    axes[0, 0].boxplot([overall_stats['psnr']], labels=['PSNR'])
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_title(f"PSNR: {stats_summary['psnr']['mean']:.2f} ± {stats_summary['psnr']['std']:.2f} dB")
    axes[0, 0].grid(True, alpha=0.3)
    
    # SSIM
    axes[0, 1].boxplot([overall_stats['ssim']], labels=['SSIM'])
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_title(f"SSIM: {stats_summary['ssim']['mean']:.4f} ± {stats_summary['ssim']['std']:.4f}")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Organ MAE
    organ_mae_data = []
    organ_labels = []
    for organ_name, data in organ_stats.items():
        if len(data['abs_diff']) > 0:
            organ_mae_data.append(data['abs_diff'])
            organ_labels.append(organ_name)
    
    if organ_mae_data:
        axes[1, 0].boxplot(organ_mae_data, labels=organ_labels)
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Organ-specific MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Contrast improvement
    if len(contrast_stats['improvement']) > 0:
        axes[1, 1].bar(['NC', 'Enhanced', 'CE (GT)'],
                      [np.mean(contrast_stats['nc_contrast']),
                       np.mean(contrast_stats['enhanced_contrast']),
                       np.mean(contrast_stats['ce_contrast'])],
                      yerr=[np.std(contrast_stats['nc_contrast']),
                            np.std(contrast_stats['enhanced_contrast']),
                            np.std(contrast_stats['ce_contrast'])],
                      color=['blue', 'green', 'red'],
                      alpha=0.7)
        axes[1, 1].set_ylabel('Contrast (Bright - Dark)')
        axes[1, 1].set_title('Liver Contrast Analysis')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statistics_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 통계 플롯 저장: {output_dir / 'statistics_plots.png'}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test 데이터셋 평가'
    )
    
    parser.add_argument('--nc-dir', type=str, required=True,
                       help='NC 데이터 디렉토리')
    parser.add_argument('--ce-dir', type=str, required=True,
                       help='CE 데이터 디렉토리 (Ground Truth)')
    parser.add_argument('--seg-dir', type=str, required=True,
                       help='Segmentation 디렉토리')
    parser.add_argument('--weight-dir', type=str, required=True,
                       help='Weight map 디렉토리')
    parser.add_argument('--split-info', type=str,
                       default=r'E:\LD-CT SR\Data\split_info.json',
                       help='split_info.json 경로 (Test set 필터링)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='평가할 split')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Phase 2 모델 체크포인트')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='결과 저장 디렉토리')
    
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--visualize-all', action='store_true',
                       help='모든 환자 시각화 (기본: 10명)')
    parser.add_argument('--visualize-count', type=int, default=10)
    parser.add_argument('--patient-ids', type=str, nargs='+',
                       help='특정 환자만 평가')
    
    args = parser.parse_args()
    
    nc_base = Path(args.nc_dir)
    ce_base = Path(args.ce_dir)
    seg_base = Path(args.seg_dir)
    weight_base = Path(args.weight_dir)
    output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / 'visualizations'
    sample_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("🧪 Test 데이터셋 평가")
    print("="*80)
    print(f"NC: {nc_base}")
    print(f"CE (GT): {ce_base}")
    print(f"Segmentation: {seg_base}")
    print(f"Weight Map: {weight_base}")
    print(f"모델: {args.checkpoint}")
    print(f"출력: {output_dir}")
    print("="*80)
    
    # 모델 로드
    print("\n모델 로딩...")
    model = TestModel(
        checkpoint_path=args.checkpoint,
        device=args.device,
        base_channels=args.base_channels
    )
    
    # 환자 목록
    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        # Load from split_info if available
        if args.split_info and Path(args.split_info).exists():
            import json
            with open(args.split_info, 'r') as f:
                split_info = json.load(f)
            
            patient_ids = split_info[args.split]['ids']
            print(f"Split info 로드: {args.split} set ({len(patient_ids)}명)")
        else:
            patient_dirs = sorted([p for p in nc_base.iterdir() if p.is_dir()])
            patient_ids = [p.name for p in patient_dirs]
            print(f"⚠️ split_info 없음 - 전체 데이터 사용")
    
    print(f"\n총 환자: {len(patient_ids)}명 ({args.split} set)")
    
    # 평가
    all_results = []
    success_count = 0
    fail_count = 0
    visualize_count = 0
    
    pbar = tqdm(patient_ids, desc="Evaluating")
    for patient_id in pbar:
        # 경로
        nc_path = nc_base / patient_id / 'NC_norm.nii.gz'
        ce_path = ce_base / patient_id / 'CE_norm.nii.gz'
        aorta_seg_path = seg_base / patient_id / 'Aorta_seg.nii.gz'
        liver_seg_path = seg_base / patient_id / 'Liver_seg.nii.gz'
        weight_map_path = weight_base / patient_id / 'NC_weight_map.nii.gz'
        
        # 존재 확인
        if not all([nc_path.exists(), ce_path.exists(),
                   aorta_seg_path.exists(), liver_seg_path.exists(),
                   weight_map_path.exists()]):
            fail_count += 1
            pbar.set_postfix({'success': success_count, 'fail': fail_count})
            continue
        
        try:
            # 평가
            result, enhanced_arr = test_patient(
                patient_id=patient_id,
                nc_path=nc_path,
                ce_path=ce_path,
                aorta_seg_path=aorta_seg_path,
                liver_seg_path=liver_seg_path,
                weight_map_path=weight_map_path,
                model=model,
                output_dir=output_dir,
                save_nifti=True
            )
            
            all_results.append(result)
            success_count += 1
            
            # 시각화
            should_visualize = args.visualize_all or (visualize_count < args.visualize_count)
            
            if should_visualize:
                nc_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(nc_path)))
                ce_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(ce_path)))
                aorta_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(aorta_seg_path)))
                liver_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(liver_seg_path)))
                weight_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(weight_map_path)))
                
                visualize_test_result(
                    patient_id=patient_id,
                    nc_arr=nc_arr,
                    ce_arr=ce_arr,
                    enhanced_arr=enhanced_arr,
                    aorta_arr=aorta_arr,
                    liver_arr=liver_arr,
                    weight_arr=weight_arr,
                    results=result,
                    output_path=sample_dir / f'{patient_id}_test.png'
                )
                visualize_count += 1
            
            pbar.set_postfix({
                'success': success_count,
                'fail': fail_count,
                'psnr': f"{result['overall']['psnr']:.2f}",
                'ssim': f"{result['overall']['ssim']:.4f}"
            })
        
        except Exception as e:
            fail_count += 1
            tqdm.write(f"✗ {patient_id}: {e}")
            pbar.set_postfix({'success': success_count, 'fail': fail_count})
    
    # 통계 생성
    if len(all_results) > 0:
        generate_statistics(all_results, output_dir)
    
    # 최종 결과
    print("\n" + "="*80)
    print("✅ 평가 완료!")
    print("="*80)
    print(f"성공: {success_count}/{len(patient_ids)}")
    print(f"실패: {fail_count}/{len(patient_ids)}")
    print(f"시각화: {visualize_count}개")
    print(f"\n결과 저장: {output_dir}")
    print(f"  - Enhanced NIfTI: {output_dir / 'HCC_XXX' / 'Enhanced_NC.nii.gz'}")
    print(f"  - 통계: {output_dir / 'statistics.json'}")
    print(f"  - CSV: {output_dir / 'results.csv'}")
    print(f"  - 시각화: {sample_dir}")
    print("="*80)


if __name__ == '__main__':
    main()