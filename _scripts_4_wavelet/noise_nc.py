from pathlib import Path
import pandas as pd


def main():
    root = Path(r"E:\LD-CT SR")
    admin_dir = root / "Outputs"
    admin_dir.mkdir(parents=True, exist_ok=True)

    noise_csv = admin_dir / "noise_summary_corrected.csv"
    meta_csv = admin_dir / "dicom_metadata_nc.csv"
    out_csv = admin_dir / "slice_noise_nc.csv"

    print("=" * 80)
    print("🧩 NPS + DICOM 메타 → slice_noise_nc.csv 생성")
    print("=" * 80)
    print(f"  noise_summary_corrected : {noise_csv}")
    print(f"  dicom_metadata_nc       : {meta_csv}")
    print(f"  output                  : {out_csv}\n")

    if not noise_csv.exists():
        raise FileNotFoundError(f"noise_summary_corrected.csv not found: {noise_csv}")
    if not meta_csv.exists():
        raise FileNotFoundError(f"dicom_metadata_nc.csv not found: {meta_csv}")

    # 1) CSV 로드
    noise = pd.read_csv(noise_csv)
    meta = pd.read_csv(meta_csv)

    print(f"  noise rows: {len(noise)}")
    print(f"  meta  rows: {len(meta)}")

    # 2) (patient, slice) 기준 merge
    merged = noise.merge(
        meta,
        on=["patient", "slice"],
        how="inner",
        validate="one_to_one",
    )

    print(f"  merged rows: {len(merged)}")

    # 3) 환자별 instance_number 기준 정렬 → z-index 부여
    merged_sorted = merged.sort_values(["patient", "instance_number"]).copy()
    merged_sorted["z"] = merged_sorted.groupby("patient").cumcount()

    # 4) 학습에서 필요한 컬럼만 정리
    slice_noise = merged_sorted[["patient", "z", "noise_std", "instance_number"]].copy()

    # patient를 문자열로 통일 (NIfTI 파일명과 매칭 위해)
    slice_noise["patient"] = slice_noise["patient"].astype(str)

    # 5) 저장
    slice_noise.to_csv(out_csv, index=False)

    print("\n✅ slice_noise_nc.csv 저장 완료!")
    print(f"  → {out_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
