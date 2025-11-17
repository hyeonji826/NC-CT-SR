# E:\LD-CT SR\_scripts_4_wavelet\check_data.py

import os
import nibabel as nib
import numpy as np
from pathlib import Path
from collections import defaultdict

print("="*80)
print("📊 STEP 1: 외부 데이터 (Low-Dose ↔ Full-Dose) 매칭 확인")
print("="*80)

# 경로 설정
low_dose_dir = Path(r"E:\LD-CT SR\Data\Images_low_dose")
full_dose_dir = Path(r"E:\LD-CT SR\Data\Images_full_dose")

# 파일 리스트 수집
low_files = sorted(list(low_dose_dir.glob("*.nii.gz")))
full_files = sorted(list(full_dose_dir.glob("*.nii.gz")))

print(f"\n✅ Low-Dose 파일 개수: {len(low_files)}")
print(f"✅ Full-Dose 파일 개수: {len(full_files)}")

# 파일명 추출 (L006_0000.nii.gz -> L006)
low_ids = set([f.stem.split('_')[0] for f in low_files])
full_ids = set([f.stem.split('_')[0] for f in full_files])

print(f"\n✅ Low-Dose 환자 ID: {sorted(low_ids)[:5]}... (총 {len(low_ids)}명)")
print(f"✅ Full-Dose 환자 ID: {sorted(full_ids)[:5]}... (총 {len(full_ids)}명)")

# 매칭 확인
matched = low_ids & full_ids
only_low = low_ids - full_ids
only_full = full_ids - low_ids

print(f"\n🎯 매칭된 환자: {len(matched)}명")
if only_low:
    print(f"⚠️  Low-Dose만 있는 환자: {sorted(only_low)}")
if only_full:
    print(f"⚠️  Full-Dose만 있는 환자: {sorted(only_full)}")

print("\n" + "="*80)
print("📊 STEP 2: 외부 데이터 품질 체크 (첫 3개 환자 샘플)")
print("="*80)

results = []

for patient_id in sorted(matched)[:3]:  # 첫 3명만 체크
    low_file = low_dose_dir / f"{patient_id}_0000.nii.gz"
    full_file = full_dose_dir / f"{patient_id}_0000.nii.gz"
    
    if not low_file.exists() or not full_file.exists():
        continue
    
    # NIfTI 로드
    low_nii = nib.load(str(low_file))
    full_nii = nib.load(str(full_file))
    
    low_data = low_nii.get_fdata()
    full_data = full_nii.get_fdata()
    
    # 정보 수집
    result = {
        'ID': patient_id,
        'Low_Shape': low_data.shape,
        'Full_Shape': full_data.shape,
        'Low_Spacing': low_nii.header.get_zooms(),
        'Full_Spacing': full_nii.header.get_zooms(),
        'Low_HU_Range': (low_data.min(), low_data.max()),
        'Full_HU_Range': (full_data.min(), full_data.max()),
        'Low_Mean': low_data.mean(),
        'Full_Mean': full_data.mean(),
    }
    results.append(result)
    
    print(f"\n🔍 환자 {patient_id}")
    print(f"  Shape       : Low {low_data.shape} | Full {full_data.shape} | Match: {low_data.shape == full_data.shape}")
    print(f"  Spacing     : Low {low_nii.header.get_zooms()} | Full {full_nii.header.get_zooms()}")
    print(f"  HU Range    : Low [{low_data.min():.1f}, {low_data.max():.1f}] | Full [{full_data.min():.1f}, {full_data.max():.1f}]")
    print(f"  Mean HU     : Low {low_data.mean():.1f} | Full {full_data.mean():.1f}")

# 전체 통계
print("\n" + "="*80)
print("📊 종합 체크")
print("="*80)

all_shapes_match = all(r['Low_Shape'] == r['Full_Shape'] for r in results)
all_spacings_match = all(r['Low_Spacing'] == r['Full_Spacing'] for r in results)

print(f"✅ 모든 Shape 일치: {all_shapes_match}")
print(f"✅ 모든 Spacing 일치: {all_spacings_match}")

print("\n" + "="*80)
print("📊 STEP 3: NC-CT 데이터 확인")
print("="*80)

nc_dir = Path(r"E:\LD-CT SR\Data\Image_NC-CT")
nc_patients = list(nc_dir.glob("*.nii.gz"))

print(f"\n✅ NC-CT 환자 수: {len(nc_patients)}명")

# 첫 3명 샘플 체크
for nc_file in nc_patients[:3]:
    patient_id = nc_file.parent.name
    nc_nii = nib.load(str(nc_file))
    nc_data = nc_nii.get_fdata()
    
    print(f"\n🔍 NC 환자 {patient_id}")
    print(f"  Shape       : {nc_data.shape}")
    print(f"  Spacing     : {nc_nii.header.get_zooms()}")
    print(f"  Value Range : [{nc_data.min():.4f}, {nc_data.max():.4f}]")
    print(f"  Mean        : {nc_data.mean():.4f}")
    print(f"  Std         : {nc_data.std():.4f}")
    
    # 정규화 방식 추정
    if nc_data.min() >= 0 and nc_data.max() <= 1.1:
        print(f"  🎯 추정 정규화: [0, 1] Min-Max")
    elif nc_data.min() >= -1.1 and nc_data.max() <= 1.1:
        print(f"  🎯 추정 정규화: [-1, 1] Standardization")
    else:
        print(f"  🎯 추정 정규화: HU 값 (정규화 안됨)")

print("\n" + "="*80)
print("✅ 데이터 체크 완료!")
print("="*80)