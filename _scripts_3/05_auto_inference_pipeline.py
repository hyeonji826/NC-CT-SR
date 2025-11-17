#!/usr/bin/env python3
"""
05_auto_inference_pipeline.py
완전 자동 추론 파이프라인

DICOM 입력 → Enhanced NC 출력 (One-Click!)

워크플로우:
1. DICOM → NIfTI 변환 및 전처리 (정규화)
2. TotalSegmentator → Aorta + Liver 세그멘테이션
3. Weight Map 생성 (간 bright/dark 구분)
4. Phase 2 모델 추론 (Seg-Guided Enhancement)
5. Enhanced NC 저장 및 시각화
"""

import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import argparse
import subprocess
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import sys

sys.path.insert(0, str(Path(__file__).parent))
from models import StructurePreservingStyleTransfer


# ============================================================
# Step 1: DICOM → NIfTI + Preprocessing
# ============================================================

def dicom_to_nifti_and_preprocess(dicom_dir, output_path, window_center=-50, window_width=400):
    """
    DICOM → NIfTI 변환 + HU windowing + 정규화
    
    Args:
        dicom_dir: DICOM 파일 폴더
        output_path: 출력 NIfTI 경로
        window_center: HU window center (간: -50)
        window_width: HU window width (간: 400)
    
    Returns:
        success: bool
        image: SimpleITK Image (성공 시)
    """
    try:
        # DICOM 읽기
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        
        if len(dicom_names) == 0:
            return False, None
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # HU 값 추출
        arr = sitk.GetArrayFromImage(image)
        
        # Windowing
        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2
        
        arr_windowed = np.clip(arr, window_min, window_max)
        
        # 정규화 [0, 1]
        arr_norm = (arr_windowed - window_min) / (window_max - window_min)
        arr_norm = arr_norm.astype(np.float32)
        
        # NIfTI 저장
        image_norm = sitk.GetImageFromArray(arr_norm)
        image_norm.CopyInformation(image)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image_norm, str(output_path))
        
        return True, image_norm
    
    except Exception as e:
        print(f"  ✗ DICOM 변환 실패: {e}")
        return False, None


# ============================================================
# Step 2: TotalSegmentator
# ============================================================

def run_totalsegmentator(dicom_dir, output_dir):
    """
    TotalSegmentator 실행 (원본 DICOM에서)
    
    Returns:
        success: bool
        aorta_path: Path or None
        liver_path: Path or None
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'TotalSegmentator',
        '-i', str(dicom_dir),
        '-o', str(output_dir),
        '--fast',
        '--roi_subset', 'aorta', 'liver'
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        
        aorta_path = output_dir / 'aorta.nii.gz'
        liver_path = output_dir / 'liver.nii.gz'
        
        if aorta_path.exists() and liver_path.exists():
            return True, aorta_path, liver_path
        else:
            return False, None, None
    
    except Exception as e:
        print(f"  ✗ TotalSegmentator 실패: {e}")
        return False, None, None


def resample_seg_to_reference(seg_image, reference_image):
    """
    Segmentation을 reference 공간으로 리샘플링
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(seg_image.GetPixelID())
    
    return resampler.Execute(seg_image)


# ============================================================
# Step 3: Weight Map 생성
# ============================================================

def create_weight_map_from_seg(nc_image, aorta_seg, liver_seg,
                                aorta_weight=1.0,
                                liver_bright_weight=0.85,
                                liver_dark_weight=0.3,
                                background_weight=0.1,
                                dark_percentile=30):
    """
    NC + Segmentation → Weight Map
    
    Args:
        nc_image: NC 정규화 이미지 (SimpleITK)
        aorta_seg: Aorta segmentation (SimpleITK)
        liver_seg: Liver segmentation (SimpleITK)
        dark_percentile: 간에서 어두운 영역 기준 (30% = 종양 등)
    
    Returns:
        weight_map: SimpleITK Image
    """
    # To numpy
    nc_arr = sitk.GetArrayFromImage(nc_image)
    aorta_arr = sitk.GetArrayFromImage(aorta_seg)
    liver_arr = sitk.GetArrayFromImage(liver_seg)
    
    # Initialize
    weight_map = np.full_like(nc_arr, background_weight, dtype=np.float32)
    
    # Aorta (최고 가중치)
    weight_map[aorta_arr > 0] = aorta_weight
    
    # Liver (bright vs dark)
    liver_mask = (liver_arr > 0)
    
    if liver_mask.sum() > 0:
        liver_intensity = nc_arr[liver_mask]
        threshold = np.percentile(liver_intensity, dark_percentile)
        
        # Bright liver (정상)
        bright_liver = liver_mask & (nc_arr > threshold)
        weight_map[bright_liver] = liver_bright_weight
        
        # Dark liver (종양)
        dark_liver = liver_mask & (nc_arr <= threshold)
        weight_map[dark_liver] = liver_dark_weight
    
    # To SimpleITK
    weight_image = sitk.GetImageFromArray(weight_map)
    weight_image.CopyInformation(nc_image)
    
    return weight_image


# ============================================================
# Step 4: 모델 추론
# ============================================================

class EnhancementModel:
    """Phase 2 모델 추론 클래스"""
    
    def __init__(self, checkpoint_path, device='cuda', base_channels=64):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = StructurePreservingStyleTransfer(base_channels=base_channels)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 모델 로드: {checkpoint_path}")
        print(f"   Device: {self.device}")
    
    @torch.no_grad()
    def enhance_nc(self, nc_image, weight_map, ce_reference=None, style_alpha=1.0):
        """
        NC 이미지 향상
        
        Args:
            nc_image: NC 정규화 이미지 (SimpleITK)
            weight_map: Weight map (SimpleITK)
            ce_reference: CE 참조 (없으면 NC로 대체)
            style_alpha: 스타일 강도 (0~1)
        
        Returns:
            enhanced_image: SimpleITK Image
        """
        # To numpy
        nc_arr = sitk.GetArrayFromImage(nc_image)
        
        # CE reference (없으면 NC 사용)
        if ce_reference is None:
            ce_arr = nc_arr.copy()
        else:
            ce_arr = sitk.GetArrayFromImage(ce_reference)
        
        # To tensor [1, 1, D, H, W]
        nc_tensor = torch.from_numpy(nc_arr).float().unsqueeze(0).unsqueeze(0).to(self.device)
        ce_tensor = torch.from_numpy(ce_arr).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Slice-by-slice 처리 (메모리 효율)
        num_slices = nc_arr.shape[0]
        enhanced_slices = []
        
        for i in tqdm(range(num_slices), desc="  Enhancing slices", leave=False):
            nc_slice = nc_tensor[:, :, i:i+1, :, :].squeeze(2)  # [1, 1, H, W]
            ce_slice = ce_tensor[:, :, i:i+1, :, :].squeeze(2)
            
            # 모델 추론
            enhanced_slice = self.model(nc_slice, ce_slice, alpha=style_alpha)
            
            # Clip to [0, 1]
            enhanced_slice = torch.clamp(enhanced_slice, 0, 1)
            
            enhanced_slices.append(enhanced_slice.cpu().numpy()[0, 0])
        
        # Stack
        enhanced_arr = np.stack(enhanced_slices, axis=0)
        
        # To SimpleITK
        enhanced_image = sitk.GetImageFromArray(enhanced_arr)
        enhanced_image.CopyInformation(nc_image)
        
        return enhanced_image


# ============================================================
# Step 5: 시각화
# ============================================================

def visualize_results(nc_path, enhanced_path, weight_path, aorta_seg_path, liver_seg_path,
                     output_path, patient_id, num_samples=6):
    """
    결과 시각화
    """
    # Load
    nc_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(nc_path)))
    enhanced_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(enhanced_path)))
    weight_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(weight_path)))
    aorta_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(aorta_seg_path)))
    liver_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(liver_seg_path)))
    
    # Select interesting slices (with organs)
    num_slices = nc_arr.shape[0]
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
    
    # Metrics
    psnr_vals = []
    ssim_vals = []
    
    for slice_idx in selected_slices:
        nc_slice = nc_arr[slice_idx]
        enh_slice = enhanced_arr[slice_idx]
        
        psnr_vals.append(psnr_metric(nc_slice, enh_slice, data_range=1.0))
        ssim_vals.append(ssim_metric(nc_slice, enh_slice, data_range=1.0))
    
    avg_psnr = np.mean(psnr_vals)
    avg_ssim = np.mean(ssim_vals)
    
    # Plot
    fig, axes = plt.subplots(5, num_samples, figsize=(4*num_samples, 20))
    fig.suptitle(
        f'Patient {patient_id} - Auto Enhancement Pipeline\n'
        f'PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}',
        fontsize=16,
        fontweight='bold'
    )
    
    for col, slice_idx in enumerate(selected_slices):
        # Row 1: Original NC
        axes[0, col].imshow(nc_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title('Original NC', fontweight='bold', color='blue')
        axes[0, col].axis('off')
        
        # Row 2: Segmentation
        axes[1, col].imshow(nc_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        aorta_mask = np.ma.masked_where(aorta_arr[slice_idx] == 0, aorta_arr[slice_idx])
        liver_mask = np.ma.masked_where(liver_arr[slice_idx] == 0, liver_arr[slice_idx])
        axes[1, col].imshow(aorta_mask, cmap='Reds', alpha=0.5)
        axes[1, col].imshow(liver_mask, cmap='Greens', alpha=0.5)
        axes[1, col].set_title('Segmentation\n(Aorta+Liver)', color='purple')
        axes[1, col].axis('off')
        
        # Row 3: Weight Map
        im = axes[2, col].imshow(weight_arr[slice_idx], cmap='jet', vmin=0.1, vmax=1.0)
        axes[2, col].set_title(f'Weight Map\n(tumor={weight_arr[slice_idx].min():.2f})', color='orange')
        axes[2, col].axis('off')
        
        # Row 4: Enhanced NC
        axes[3, col].imshow(enhanced_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[3, col].set_title(f'Enhanced NC\nPSNR: {psnr_vals[col]:.2f}',
                              fontweight='bold', color='green')
        axes[3, col].axis('off')
        
        # Row 5: Difference
        diff = np.abs(enhanced_arr[slice_idx] - nc_arr[slice_idx])
        axes[4, col].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[4, col].set_title(f'Change\nSSIM: {ssim_vals[col]:.4f}')
        axes[4, col].axis('off')
    
    plt.colorbar(im, ax=axes[2, :], fraction=0.046, pad=0.04)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ 시각화 저장: {output_path.name}")


# ============================================================
# Main Pipeline
# ============================================================

def process_patient(patient_id, dicom_dir, model, output_dir, temp_dir, 
                   visualize=True, sample_dir=None):
    """
    환자 1명 완전 자동 처리
    
    Returns:
        success: bool
        message: str
    """
    print(f"\n{'='*80}")
    print(f"Patient: {patient_id}")
    print(f"{'='*80}")
    
    patient_output_dir = output_dir / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_temp_dir = temp_dir / patient_id
    patient_temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: DICOM → NIfTI + Preprocessing
        print("[1/5] DICOM → NIfTI + Preprocessing...")
        nc_norm_path = patient_output_dir / 'NC_norm.nii.gz'
        
        success, nc_image = dicom_to_nifti_and_preprocess(
            dicom_dir=dicom_dir,
            output_path=nc_norm_path
        )
        
        if not success:
            return False, "DICOM conversion failed"
        
        print(f"  ✅ NC_norm.nii.gz 저장")
        
        # Step 2: TotalSegmentator
        print("[2/5] TotalSegmentator (Aorta + Liver)...")
        seg_temp_dir = patient_temp_dir / 'segmentation'
        
        success, aorta_orig_path, liver_orig_path = run_totalsegmentator(
            dicom_dir=dicom_dir,
            output_dir=seg_temp_dir
        )
        
        if not success:
            return False, "TotalSegmentator failed"
        
        # Resample to NC space
        aorta_orig = sitk.ReadImage(str(aorta_orig_path))
        liver_orig = sitk.ReadImage(str(liver_orig_path))
        
        aorta_resampled = resample_seg_to_reference(aorta_orig, nc_image)
        liver_resampled = resample_seg_to_reference(liver_orig, nc_image)
        
        aorta_seg_path = patient_output_dir / 'Aorta_seg.nii.gz'
        liver_seg_path = patient_output_dir / 'Liver_seg.nii.gz'
        
        sitk.WriteImage(aorta_resampled, str(aorta_seg_path))
        sitk.WriteImage(liver_resampled, str(liver_seg_path))
        
        print(f"  ✅ Segmentation 저장")
        
        # Step 3: Weight Map
        print("[3/5] Weight Map 생성...")
        weight_map = create_weight_map_from_seg(
            nc_image=nc_image,
            aorta_seg=aorta_resampled,
            liver_seg=liver_resampled
        )
        
        weight_map_path = patient_output_dir / 'NC_weight_map.nii.gz'
        sitk.WriteImage(weight_map, str(weight_map_path))
        
        print(f"  ✅ Weight Map 저장")
        
        # Step 4: 모델 추론
        print("[4/5] Phase 2 모델 추론...")
        enhanced_image = model.enhance_nc(
            nc_image=nc_image,
            weight_map=weight_map,
            ce_reference=None,
            style_alpha=1.0
        )
        
        enhanced_path = patient_output_dir / 'Enhanced_NC.nii.gz'
        sitk.WriteImage(enhanced_image, str(enhanced_path))
        
        print(f"  ✅ Enhanced NC 저장")
        
        # Step 5: 시각화
        if visualize and sample_dir:
            print("[5/5] 시각화...")
            visualize_results(
                nc_path=nc_norm_path,
                enhanced_path=enhanced_path,
                weight_path=weight_map_path,
                aorta_seg_path=aorta_seg_path,
                liver_seg_path=liver_seg_path,
                output_path=sample_dir / f'{patient_id}_result.png',
                patient_id=patient_id
            )
        
        # Cleanup temp
        shutil.rmtree(patient_temp_dir)
        
        return True, "Success"
    
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description='완전 자동 추론 파이프라인: DICOM → Enhanced NC'
    )
    
    parser.add_argument('--dicom-dir', type=str, required=True,
                       help='DICOM 폴더 (환자별 서브폴더)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Phase 2 모델 체크포인트')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='출력 디렉토리')
    parser.add_argument('--sample-dir', type=str,
                       help='시각화 샘플 저장 디렉토리')
    parser.add_argument('--temp-dir', type=str, default='./temp',
                       help='임시 디렉토리')
    
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--visualize-samples', type=int, default=10,
                       help='시각화할 환자 수')
    parser.add_argument('--patient-ids', type=str, nargs='+',
                       help='처리할 환자 ID (없으면 전체)')
    
    args = parser.parse_args()
    
    dicom_base = Path(args.dicom_dir)
    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    if args.sample_dir:
        sample_dir = Path(args.sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
    else:
        sample_dir = None
    
    print("="*80)
    print("🚀 완전 자동 추론 파이프라인")
    print("="*80)
    print(f"DICOM: {dicom_base}")
    print(f"출력: {output_dir}")
    print(f"모델: {args.checkpoint}")
    print("\n워크플로우:")
    print("  1. DICOM → NIfTI + Preprocessing")
    print("  2. TotalSegmentator (Aorta + Liver)")
    print("  3. Weight Map 생성")
    print("  4. Phase 2 모델 추론")
    print("  5. Enhanced NC 저장 + 시각화")
    print("="*80)
    
    # 모델 로드
    print("\n모델 로딩...")
    model = EnhancementModel(
        checkpoint_path=args.checkpoint,
        device=args.device,
        base_channels=args.base_channels
    )
    
    # 환자 목록
    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_dirs = sorted([p for p in dicom_base.iterdir() if p.is_dir()])
        patient_ids = [p.name for p in patient_dirs]
    
    total_patients = len(patient_ids)
    print(f"\n총 환자: {total_patients}명")
    
    success_count = 0
    fail_count = 0
    visualize_count = 0
    failed_patients = []
    
    for patient_id in patient_ids:
        dicom_patient_dir = dicom_base / patient_id
        
        if not dicom_patient_dir.exists():
            fail_count += 1
            failed_patients.append((patient_id, "DICOM folder not found"))
            continue
        
        # 이미 처리됨?
        enhanced_path = output_dir / patient_id / 'Enhanced_NC.nii.gz'
        if enhanced_path.exists():
            print(f"\n✓ {patient_id}: Already processed (skip)")
            success_count += 1
            continue
        
        # 처리
        should_visualize = visualize_count < args.visualize_samples
        
        success, message = process_patient(
            patient_id=patient_id,
            dicom_dir=dicom_patient_dir,
            model=model,
            output_dir=output_dir,
            temp_dir=temp_dir,
            visualize=should_visualize,
            sample_dir=sample_dir
        )
        
        if success:
            success_count += 1
            if should_visualize:
                visualize_count += 1
            print(f"✅ {patient_id}: Success")
        else:
            fail_count += 1
            failed_patients.append((patient_id, message))
            print(f"✗ {patient_id}: {message}")
    
    # 결과
    print("\n" + "="*80)
    print("완료!")
    print("="*80)
    print(f"성공: {success_count}/{total_patients}")
    print(f"실패: {fail_count}/{total_patients}")
    
    if failed_patients:
        print(f"\n실패한 환자:")
        for pid, msg in failed_patients:
            print(f"  - {pid}: {msg}")
    
    print(f"\n출력: {output_dir}")
    if sample_dir:
        print(f"샘플: {sample_dir}")
    print("="*80)


if __name__ == '__main__':
    main()