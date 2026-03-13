"""
Batch Streak Map Generation
============================
전체 NC-CT 환자에 대해 streak_map을 생성하고 저장.

출력:
  {output_root}/{patient_id}.npy — (Z, H, W) float32 streak intensity (HU)

실행:
  python batch_streak_maps.py
  python batch_streak_maps.py --patient 0104710   # 단일 환자
"""

import os
import sys
import argparse
import csv
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streak_removal_3d import load_volume, detect_streaks_3d


def load_exposure_map(csv_path):
    """DICOM metadata CSV에서 {patient_id: exposure_mAs} dict 생성."""
    exposure = {}
    if not os.path.exists(csv_path):
        print(f"[WARN] Metadata CSV not found: {csv_path}")
        return exposure
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['PatientID']
            try:
                exposure[pid] = int(row['Exposure_mAs'])
            except (ValueError, KeyError):
                pass
    print(f"Loaded exposure data for {len(exposure)} patients")
    return exposure


def main():
    parser = argparse.ArgumentParser(description='Batch Streak Map Generation')
    parser.add_argument('--nifti_dir',
                        default='F:/LD-CT SR/Data/NC-CT NIfTI')
    parser.add_argument('--output_dir',
                        default='F:/LD-CT SR/Data/NC-CT Streak Maps')
    parser.add_argument('--metadata_csv',
                        default='F:/LD-CT SR/Outputs/streak_removal_3d/dicom_metadata_all.csv')
    parser.add_argument('--patient', default=None,
                        help='Single patient ID (skip others)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing streak maps')
    parser.add_argument('--kernel_z', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Exposure data
    exposure_map = load_exposure_map(args.metadata_csv)

    # NIfTI files
    nifti_dir = Path(args.nifti_dir)
    nifti_files = sorted(nifti_dir.glob('*.nii*'))
    print(f"Found {len(nifti_files)} NIfTI files")

    # Filter single patient
    if args.patient:
        nifti_files = [f for f in nifti_files if args.patient in f.stem]
        if not nifti_files:
            print(f"Patient '{args.patient}' not found!")
            return

    done = 0
    skipped = 0
    failed = 0

    for i, nifti_path in enumerate(nifti_files):
        # Patient ID
        stem = nifti_path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]  # handle .nii.gz double extension
        patient_id = stem

        out_path = os.path.join(args.output_dir, f"{patient_id}.npy")

        # Skip existing
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(nifti_files)}] {patient_id}")

        try:
            # Load
            volume, affine, header = load_volume(str(nifti_path))
            H, W, Z = volume.shape

            # Exposure
            exposure_mAs = exposure_map.get(patient_id)
            if exposure_mAs:
                print(f"  Exposure: {exposure_mAs} mAs")

            # Detect streaks
            streak_map, anatomy_z, body_3d = detect_streaks_3d(
                volume,
                kernel_z=args.kernel_z,
                exposure_mAs=exposure_mAs,
                verbose=True,
            )

            # Save as (Z, H, W) to match dataset convention
            streak_zhw = np.transpose(streak_map, (2, 0, 1))  # (H,W,Z) → (Z,H,W)
            np.save(out_path, streak_zhw.astype(np.float32))

            body_pct = 100 * body_3d.sum() / body_3d.size
            body_stk = streak_map[body_3d]
            print(f"  Saved: {out_path}")
            print(f"  Shape: {streak_zhw.shape}, Body: {body_pct:.0f}%")
            print(f"  Streak range: [{body_stk.min():.1f}, {body_stk.max():.1f}] HU")
            done += 1

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n=== Done ===")
    print(f"  Processed: {done}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output dir: {args.output_dir}")


if __name__ == '__main__':
    main()
