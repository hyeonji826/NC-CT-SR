"""
원본 DICOM (HU 값) → TotalSegmentator → Segmentation (원본 공간)
                                              ↓
                                        리샘플링 (NC_norm 공간으로)
                                              ↓
                                    Aorta_seg.nii.gz, Liver_seg.nii.gz
"""

import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import shutil


def run_totalsegmentator_on_dicom(dicom_dir, output_dir):
    """
    원본 DICOM 폴더에서 TotalSegmentator 실행
    
    Args:
        dicom_dir: DICOM 파일들이 있는 폴더
        output_dir: 출력 디렉토리
    
    Returns:
        success: bool
        error: str or None
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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5분 타임아웃
        )
        return True, None
    except subprocess.TimeoutExpired:
        return False, "Timeout (5 minutes)"
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"
    except Exception as e:
        return False, str(e)


def resample_to_reference(moving_image, reference_image, is_label=True):
    """
    moving_image를 reference_image와 같은 공간으로 리샘플링
    
    Args:
        moving_image: 리샘플링할 이미지 (segmentation mask)
        reference_image: 기준 이미지 (NC_norm.nii.gz)
        is_label: True면 nearest neighbor (segmentation용)
    
    Returns:
        resampled_image: SimpleITK Image
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    
    if is_label:
        # Segmentation mask: nearest neighbor
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # Intensity image: linear
        resampler.SetInterpolator(sitk.sitkLinear)
    
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(moving_image.GetPixelID())
    
    resampled = resampler.Execute(moving_image)
    return resampled


def process_patient(patient_id, dicom_dir, nc_norm_path, output_dir, sample_dir, visualize=True):
    """
    환자 1명 처리
    
    Args:
        patient_id: 환자 ID
        dicom_dir: 원본 DICOM 폴더
        nc_norm_path: NC_norm.nii.gz 경로 (기준)
        output_dir: 출력 디렉토리
        sample_dir: 샘플 저장 디렉토리
        visualize: 시각화 여부
    
    Returns:
        success: bool
        message: str
    """
    # 1. NC_norm 로드 (기준 공간)
    if not nc_norm_path.exists():
        return False, "NC_norm.nii.gz not found"
    
    nc_ref = sitk.ReadImage(str(nc_norm_path))
    
    # 2. TotalSegmentator 실행 (임시 폴더)
    temp_seg_dir = output_dir / f'temp_seg_{patient_id}'
    
    print(f"  [1/4] Running TotalSegmentator on DICOM...")
    success, error = run_totalsegmentator_on_dicom(dicom_dir, temp_seg_dir)
    
    if not success:
        if temp_seg_dir.exists():
            shutil.rmtree(temp_seg_dir)
        return False, f"TotalSegmentator failed: {error}"
    
    # 3. Segmentation 결과 로드
    aorta_seg_path = temp_seg_dir / 'aorta.nii.gz'
    liver_seg_path = temp_seg_dir / 'liver.nii.gz'
    
    if not aorta_seg_path.exists() or not liver_seg_path.exists():
        if temp_seg_dir.exists():
            shutil.rmtree(temp_seg_dir)
        return False, "Segmentation output files not found"
    
    print(f"  [2/4] Loading segmentation results...")
    aorta_seg = sitk.ReadImage(str(aorta_seg_path))
    liver_seg = sitk.ReadImage(str(liver_seg_path))
    
    # 4. NC_norm 공간으로 리샘플링
    print(f"  [3/4] Resampling to NC_norm space...")
    aorta_resampled = resample_to_reference(aorta_seg, nc_ref, is_label=True)
    liver_resampled = resample_to_reference(liver_seg, nc_ref, is_label=True)
    
    # 5. 저장
    final_output_dir = output_dir / patient_id
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    aorta_output_path = final_output_dir / 'Aorta_seg.nii.gz'
    liver_output_path = final_output_dir / 'Liver_seg.nii.gz'
    
    sitk.WriteImage(aorta_resampled, str(aorta_output_path))
    sitk.WriteImage(liver_resampled, str(liver_output_path))
    
    print(f"  [4/4] Saved: {aorta_output_path.name}, {liver_output_path.name}")
    
    # 임시 폴더 삭제
    shutil.rmtree(temp_seg_dir)
    
    # 6. 시각화
    if visualize:
        try:
            visualize_segmentation(
                nc_path=nc_norm_path,
                aorta_path=aorta_output_path,
                liver_path=liver_output_path,
                output_path=sample_dir / f'{patient_id}_segmentation.png',
                patient_id=patient_id
            )
        except Exception as e:
            return True, f"Segmentation OK, but visualization failed: {e}"
    
    return True, "Success"


def visualize_segmentation(nc_path, aorta_path, liver_path, output_path, patient_id, num_samples=4):
    """
    Segmentation 결과 시각화
    """
    # Load
    nc_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(nc_path)))
    aorta_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(aorta_path)))
    liver_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(liver_path)))
    
    # Select slices with organs
    num_slices = nc_arr.shape[0]
    valid_slices = []
    
    for i in range(num_slices):
        aorta_pixels = aorta_arr[i].sum()
        liver_pixels = liver_arr[i].sum()
        
        if aorta_pixels > 10 and liver_pixels > 100:
            valid_slices.append((i, aorta_pixels + liver_pixels))
    
    if len(valid_slices) < num_samples:
        # Fallback: liver만 있어도 OK
        valid_slices = []
        for i in range(num_slices):
            if liver_arr[i].sum() > 100:
                valid_slices.append((i, liver_arr[i].sum()))
    
    # Sort by pixel count and select top N
    valid_slices.sort(key=lambda x: x[1], reverse=True)
    selected_slices = [s[0] for s in valid_slices[:num_samples]]
    
    if len(selected_slices) < num_samples:
        # 최후의 fallback
        selected_slices = np.linspace(
            int(num_slices * 0.3),
            int(num_slices * 0.7),
            num_samples,
            dtype=int
        ).tolist()
    
    # Plot
    fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
    fig.suptitle(
        f'Patient {patient_id} - Segmentation from Original DICOM\n'
        f'✅ Aorta (Red) + Liver (Green)',
        fontsize=16,
        fontweight='bold'
    )
    
    for col, slice_idx in enumerate(selected_slices):
        # NC
        axes[0, col].imshow(nc_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'Slice {slice_idx}: NC', fontsize=10, fontweight='bold')
        axes[0, col].axis('off')
        
        # Aorta
        axes[1, col].imshow(nc_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        aorta_mask = np.ma.masked_where(aorta_arr[slice_idx] == 0, aorta_arr[slice_idx])
        axes[1, col].imshow(aorta_mask, cmap='Reds', alpha=0.7, vmin=0, vmax=1)
        aorta_count = int(aorta_arr[slice_idx].sum())
        axes[1, col].set_title(f'Aorta\n({aorta_count} pixels)', 
                              fontsize=10, color='red', fontweight='bold')
        axes[1, col].axis('off')
        
        # Liver
        axes[2, col].imshow(nc_arr[slice_idx], cmap='gray', vmin=0, vmax=1)
        liver_mask = np.ma.masked_where(liver_arr[slice_idx] == 0, liver_arr[slice_idx])
        axes[2, col].imshow(liver_mask, cmap='Greens', alpha=0.7, vmin=0, vmax=1)
        liver_count = int(liver_arr[slice_idx].sum())
        axes[2, col].set_title(f'Liver\n({liver_count} pixels)', 
                              fontsize=10, color='green', fontweight='bold')
        axes[2, col].axis('off')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Segmentation from Original DICOM → Resample to NC_norm space'
    )
    
    parser.add_argument('--dicom-dir', type=str,
                       default=r'E:\LD-CT SR\Data\HCC Abd NC-CT',
                       help='원본 DICOM 폴더')
    parser.add_argument('--nc-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC',
                       help='NC_norm.nii.gz 폴더 (기준)')
    parser.add_argument('--output-dir', type=str,
                       default=r'E:\LD-CT SR\Data\segmentation',
                       help='출력 디렉토리')
    parser.add_argument('--sample-dir', type=str,
                       default=r'E:\LD-CT SR\Data2\samples\segmentation',
                       help='시각화 샘플')
    parser.add_argument('--visualize-samples', type=int, default=10)
    parser.add_argument('--start-from', type=int, default=0)
    
    args = parser.parse_args()
    
    dicom_base = Path(args.dicom_dir)
    nc_base = Path(args.nc_dir)
    output_dir = Path(args.output_dir)
    sample_dir = Path(args.sample_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Segmentation from Original DICOM")
    print("="*80)
    print(f"DICOM: {dicom_base}")
    print(f"NC (reference): {nc_base}")
    print(f"출력: {output_dir}")
    print(f"샘플: {sample_dir}")
    print("\n워크플로우:")
    print("  1. 원본 DICOM → TotalSegmentator (원본 HU 값)")
    print("  2. Segmentation → NC_norm 공간으로 리샘플링")
    print("  3. 저장 및 시각화")
    print("="*80)
    
    # 환자 목록 (NC 기준)
    patient_dirs = sorted([p for p in nc_base.iterdir() if p.is_dir()])
    total_patients = len(patient_dirs)
    
    if args.start_from > 0:
        patient_dirs = patient_dirs[args.start_from:]
    
    print(f"\n총 환자: {total_patients}")
    print(f"처리할 환자: {len(patient_dirs)}")
    
    success_count = 0
    fail_count = 0
    visualize_count = 0
    failed_patients = []
    
    pbar = tqdm(patient_dirs, desc="Processing")
    for patient_dir in pbar:
        patient_id = patient_dir.name
        
        # 경로 확인
        dicom_patient_dir = dicom_base / patient_id
        nc_norm_path = patient_dir / 'NC_norm.nii.gz'
        
        if not dicom_patient_dir.exists():
            fail_count += 1
            failed_patients.append((patient_id, "DICOM folder not found"))
            pbar.set_postfix({'success': success_count, 'fail': fail_count, 'status': 'skip'})
            continue
        
        # 이미 처리됨?
        final_output_dir = output_dir / patient_id
        if (final_output_dir / 'Aorta_seg.nii.gz').exists() and \
           (final_output_dir / 'Liver_seg.nii.gz').exists():
            success_count += 1
            pbar.set_postfix({'success': success_count, 'fail': fail_count, 'status': 'skip (done)'})
            continue
        
        # 처리
        should_visualize = visualize_count < args.visualize_samples
        
        success, message = process_patient(
            patient_id=patient_id,
            dicom_dir=dicom_patient_dir,
            nc_norm_path=nc_norm_path,
            output_dir=output_dir,
            sample_dir=sample_dir,
            visualize=should_visualize
        )
        
        if success:
            success_count += 1
            if should_visualize:
                visualize_count += 1
            pbar.set_postfix({'success': success_count, 'fail': fail_count, 'status': 'ok'})
        else:
            fail_count += 1
            failed_patients.append((patient_id, message))
            tqdm.write(f"✗ {patient_id}: {message}")
            pbar.set_postfix({'success': success_count, 'fail': fail_count, 'status': 'fail'})
    
    # 결과
    print("\n" + "="*80)
    print("완료!")
    print("="*80)
    print(f"성공: {success_count}/{total_patients}")
    print(f"실패: {fail_count}/{total_patients}")
    print(f"시각화: {visualize_count}개")
    
    if failed_patients:
        print(f"\n실패한 환자: {len(failed_patients)}명")
        for pid, msg in failed_patients[:10]:
            print(f"  - {pid}: {msg}")
    
    print(f"\n출력: {output_dir}")
    print(f"샘플: {sample_dir}")
    print("="*80)


if __name__ == '__main__':
    main()