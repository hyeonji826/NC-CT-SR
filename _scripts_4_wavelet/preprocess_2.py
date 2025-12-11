"""
NC-CT DICOM → NIfTI 전처리 (Aggressive Body Cropping)

핵심: 
- Threshold -200 (soft tissue만)
- Margin 10 pixels (tight crop)
- Air 완전 제거
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("🔄 NC-CT DICOM → NIfTI 전처리 (Aggressive Body Cropping)")
print("="*80)

dicom_root = Path(r"E:\LD-CT SR\Data\HCC Abd NC-CT")
output_dir = Path(r"E:\LD-CT SR\Data\Image_NC-CT_v2")
output_dir.mkdir(exist_ok=True, parents=True)

patient_dirs = sorted([d for d in dicom_root.iterdir() if d.is_dir()])

print(f"\n✅ 발견된 환자 수: {len(patient_dirs)}명\n")

success_count = 0
fail_count = 0

for patient_dir in tqdm(patient_dirs, desc="Processing"):
    patient_id = patient_dir.name

    try:
        # DICOM series 읽기
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(patient_dir))

        if len(dicom_names) == 0:
            print(f"⚠️  {patient_id}: DICOM 파일 없음")
            fail_count += 1
            continue

        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # NumPy array로 변환
        vol = sitk.GetArrayFromImage(image)  # (Z, Y, X)
        
        # ⭐ Aggressive threshold: -200 HU (soft tissue만)
        body_mask = vol > -200
        
        # Morphological closing으로 작은 구멍 메우기
        from scipy.ndimage import binary_closing
        for z in range(body_mask.shape[0]):
            body_mask[z] = binary_closing(body_mask[z], structure=np.ones((5, 5)))
        
        # X, Y 축에서 body 영역 찾기
        y_indices = np.where(body_mask.any(axis=(0, 2)))[0]
        x_indices = np.where(body_mask.any(axis=(0, 1)))[0]
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            print(f"⚠️  {patient_id}: Body 영역 찾을 수 없음")
            fail_count += 1
            continue
        
        # ⭐ Tight crop: margin 10 pixels
        margin = 10
        y_min = max(0, y_indices[0] - margin)
        y_max = min(vol.shape[1], y_indices[-1] + 1 + margin)
        x_min = max(0, x_indices[0] - margin)
        x_max = min(vol.shape[2], x_indices[-1] + 1 + margin)
        
        # Crop
        vol_cropped = vol[:, y_min:y_max, x_min:x_max]
        
        # Air 영역을 -160 HU로 clamp (window min)
        vol_cropped = np.clip(vol_cropped, -160, 3000)
        
        # SimpleITK Image로 변환
        image_cropped = sitk.GetImageFromArray(vol_cropped)
        
        # Spacing, Origin, Direction 복사
        original_spacing = image.GetSpacing()
        original_origin = image.GetOrigin()
        original_direction = image.GetDirection()
        
        image_cropped.SetSpacing(original_spacing)
        new_origin = list(original_origin)
        new_origin[0] += x_min * original_spacing[0]
        new_origin[1] += y_min * original_spacing[1]
        image_cropped.SetOrigin(tuple(new_origin))
        image_cropped.SetDirection(original_direction)
        
        # NIfTI 저장
        output_file = output_dir / f"{patient_id}_0000.nii.gz"
        sitk.WriteImage(image_cropped, str(output_file))
        
        # 통계
        if patient_id in ["0025980", "0040386"]:
            print(f"\n  {patient_id}:")
            print(f"    Cropped HU: [{vol_cropped.min():.0f}, {vol_cropped.max():.0f}]")
            print(f"    Cropped mean: {vol_cropped.mean():.1f}")
            body_fraction = (vol_cropped > -200).sum() / vol_cropped.size
            print(f"    Body fraction: {body_fraction*100:.1f}%")

        success_count += 1

    except Exception as e:
        print(f"❌ {patient_id}: {str(e)}")
        fail_count += 1

print("\n" + "="*80)
print("✅ 전처리 완료!")
print(f"✅ 성공: {success_count}명")
print(f"❌ 실패: {fail_count}명")
print("\n⚠️  다음: verify_dataset.py 재실행")
print("="*80)