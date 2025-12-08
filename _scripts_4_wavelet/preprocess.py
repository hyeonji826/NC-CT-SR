# E:\LD-CT SR\_scripts_4_wavelet\preprocess_nc_dicom_v2.py

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("🔄 NC-CT DICOM → NIfTI 전처리 (Orientation 수정)")
print("="*80)

dicom_root = Path(r"E:\LD-CT SR\Data\HCC Abd NC-CT")
output_dir = Path(r"E:\LD-CT SR\Data\Image_NC-CT")
output_dir.mkdir(exist_ok=True, parents=True)

patient_dirs = sorted([d for d in dicom_root.iterdir() if d.is_dir()])

print(f"\n✅ 발견된 환자 수: {len(patient_dirs)}명\n")

success_count = 0
fail_count = 0

for patient_dir in tqdm(patient_dirs, desc="Processing"):
    patient_id = patient_dir.name
    
    try:
        # DICOM series 읽기 (orientation 자동 처리)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(patient_dir))
        
        if len(dicom_names) == 0:
            print(f"⚠️  {patient_id}: DICOM 파일 없음")
            fail_count += 1
            continue
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        # 현재 크기 확인
        original_size = image.GetSize()  # (X, Y, Z)
        original_spacing = image.GetSpacing()
        
        # 512×512로 resampling
        target_size = (512, 512, original_size[2])
        
        new_spacing = (
            original_spacing[0] * original_size[0] / target_size[0],
            original_spacing[1] * original_size[1] / target_size[1],
            original_spacing[2]
        )
        
        # Resampling
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())  # ← 핵심!
        resampler.SetInterpolator(sitk.sitkBSpline)
        
        resampled_image = resampler.Execute(image)
        
        # NIfTI 저장
        output_file = output_dir / f"{patient_id}_0000.nii.gz"
        sitk.WriteImage(resampled_image, str(output_file))
        
        success_count += 1
        
    except Exception as e:
        print(f"❌ {patient_id}: {str(e)}")
        fail_count += 1

print("\n" + "="*80)
print("✅ 전처리 완료!")
print(f"✅ 성공: {success_count}명")
print(f"❌ 실패: {fail_count}명")