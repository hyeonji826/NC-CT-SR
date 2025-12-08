# -*- coding: utf-8 -*-
"""
NC-CT DICOM 메타데이터 → CSV 일괄 추출 스크립트

- 입력 루트: E:\LD-CT SR\Data\HCC Abd NC-CT\{patient}\*.dcm
- 출력 CSV: E:\LD-CT SR\00_admin\dicom_metadata_nc.csv

csv 컬럼 (예시):
    patient          : 상위 폴더명 (ex. 25980)
    slice            : 파일명 (ex. CT0136.dcm)
    rel_path         : root 기준 상대경로
    series_uid       : SeriesInstanceUID
    study_uid        : StudyInstanceUID
    sop_uid          : SOPInstanceUID
    instance_number  : InstanceNumber
    image_position   : ImagePositionPatient (x,y,z) 쉼표 구분 문자열
    image_orientation: ImageOrientationPatient (6개 원소) 쉼표 구분 문자열
    pixel_spacing    : PixelSpacing (row, col)
    slice_thickness  : SliceThickness
    kvp              : kVp
    exposure         : Exposure / ExposureTime 등 (있으면)
    window_center    : WindowCenter (첫 값)
    window_width     : WindowWidth (첫 값)
"""

import csv
from pathlib import Path

import pydicom


def safe_get(ds, name, default=None):
    """DICOM 태그 safely 추출"""
    if not hasattr(ds, name):
        return default
    value = getattr(ds, name)
    # 다중 값인 경우 문자열로 합치기
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def main():
    dicom_root = Path(r"E:\LD-CT SR\Data\HCC Abd NC-CT")
    output_csv = Path(r"E:\LD-CT SR\Outputs\dicom_metadata_nc.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📝 NC-CT DICOM 메타데이터 → CSV 추출")
    print("=" * 80)
    print(f"  DICOM root : {dicom_root}")
    print(f"  Output CSV : {output_csv}\n")

    if not dicom_root.exists():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")

    # CSV 헤더
    fieldnames = [
        "patient",
        "slice",
        "rel_path",
        "series_uid",
        "study_uid",
        "sop_uid",
        "instance_number",
        "image_position",
        "image_orientation",
        "pixel_spacing",
        "slice_thickness",
        "kvp",
        "exposure",
        "window_center",
        "window_width",
    ]

    total_files = 0
    success = 0
    failed = 0

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # patient 디렉토리 단위로 순회
        patient_dirs = sorted(
            [d for d in dicom_root.iterdir() if d.is_dir()]
        )

        print(f"✅ 발견된 환자 수: {len(patient_dirs)}명\n")

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name

            # 하위 모든 DICOM 파일 탐색 (*.dcm)
            dicom_files = sorted(patient_dir.rglob("*.dcm"))
            if not dicom_files:
                print(f"⚠️  {patient_id}: DICOM 파일 없음 (스킵)")
                continue

            for dcm_path in dicom_files:
                total_files += 1
                try:
                    # PixelData는 필요 없으니 stop_before_pixels=True (속도 ↑, 메모리 ↓)
                    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)

                    row = {
                        "patient": patient_id,
                        "slice": dcm_path.name,
                        "rel_path": str(dcm_path.relative_to(dicom_root)),
                        "series_uid": safe_get(ds, "SeriesInstanceUID", ""),
                        "study_uid": safe_get(ds, "StudyInstanceUID", ""),
                        "sop_uid": safe_get(ds, "SOPInstanceUID", ""),
                        "instance_number": safe_get(ds, "InstanceNumber", ""),
                        "image_position": safe_get(ds, "ImagePositionPatient", ""),
                        "image_orientation": safe_get(ds, "ImageOrientationPatient", ""),
                        "pixel_spacing": safe_get(ds, "PixelSpacing", ""),
                        "slice_thickness": safe_get(ds, "SliceThickness", ""),
                        "kvp": safe_get(ds, "KVP", ""),
                        # Exposure 관련 태그는 장비마다 다를 수 있으니 몇 개 시도
                        "exposure": (
                            safe_get(ds, "Exposure", "")
                            or safe_get(ds, "ExposureTime", "")
                        ),
                        "window_center": safe_get(ds, "WindowCenter", ""),
                        "window_width": safe_get(ds, "WindowWidth", ""),
                    }

                    writer.writerow(row)
                    success += 1

                except Exception as e:
                    failed += 1
                    print(f"❌ {patient_id} / {dcm_path.name}: {e}")

    print("\n" + "=" * 80)
    print("✅ 메타데이터 CSV 생성 완료")
    print(f"  총 DICOM 파일 수 : {total_files}")
    print(f"  성공             : {success}")
    print(f"  실패             : {failed}")
    print(f"\n  → 결과 파일: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
