"""
NC-CT DICOM → NIfTI 전처리 (3D Resample + BSpline)

- DICOM root : E:\LD-CT SR\Data\HCC Abd NC-CT
- Output     : E:\LD-CT SR\Data\Image_NC-CT_v3\{patient}_0000.nii.gz

주의:
- target_spacing_z를 바꾸면 Z축 slice 개수가 달라질 수 있음
  → 그 경우 slice_noise_nc.csv (z 인덱스 기준)도 다시 만드는 게 안전함
"""

import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

print("="*80)
print("🔄 NC-CT DICOM → NIfTI 전처리 (3D Resample + BSpline)")
print("="*80)

dicom_root = Path(r"E:\LD-CT SR\Data\HCC Abd NC-CT")
output_dir = Path(r"E:\LD-CT SR\Data\Image_NC-CT_3d")
output_dir.mkdir(exist_ok=True, parents=True)

# 🔧 여기서 target_spacing을 한 번만 정해서 전체에 통일
# CE-CT에서 쓰는 spacing이 있다면 그 값으로 맞추는 게 제일 좋고,
# 없으면 대략적인 예시로 (0.8, 0.8, 2.0) 정도를 시작점으로 사용.
target_spacing = (0.8, 0.8, 2.0)  # (sx, sy, sz) [mm]

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

        original_size = np.array(list(image.GetSize()), dtype=float)    # (X, Y, Z)
        original_spacing = np.array(list(image.GetSpacing()), dtype=float)

        # 전체 물리 길이 [mm]
        physical_size = original_size * original_spacing

        # 3D target spacing으로 재구성할 size 계산
        target_spacing_np = np.array(target_spacing, dtype=float)
        target_size = np.rint(physical_size / target_spacing_np).astype(int)
        target_size = tuple(int(max(1, s)) for s in target_size)  # 최소 1

        # Resample 설정
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetOutputSpacing(tuple(target_spacing_np.tolist()))
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
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
print("✅ 전처리(3D Resample) 완료!")
print(f"✅ 성공: {success_count}명")
print(f"❌ 실패: {fail_count}명")
