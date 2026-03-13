"""
Pre-compute action-guided masks for all NC-CT volumes.
======================================================

저장 형식: {patient_id}.npz
  - artifact: (Z, H, W) uint8 (probability × 255)
  - fluid:    (Z, H, W) uint8
  - structure:(Z, H, W) uint8

학습 시 loss weighting에 사용:
  - artifact 영역: "여기는 반드시 제거" → L1 weight 높임
  - fluid 영역:    "여기는 절대 보존" → preservation weight 높임
  - structure 영역: "경계 살려"       → edge weight 높임

사용법:
  python preprocess_masks.py                          # config 기본 경로
  python preprocess_masks.py --input_dir ... --output_dir ...
  python preprocess_masks.py --force                  # 기존 파일 덮어쓰기
"""

import argparse
import numpy as np
import nibabel as nib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_guided_masks import generate_action_masks


def _extract_patient_id(nifti_path: Path) -> str:
    """
    NIfTI 파일명에서 patient_id 추출.
    .nii → stem 그대로
    .nii.gz → .nii.gz 모두 제거
    """
    name = nifti_path.name
    if name.endswith('.nii.gz'):
        return name[:-len('.nii.gz')]
    return nifti_path.stem


def _collect_nifti_files(input_dir: Path) -> list:
    """
    .nii와 .nii.gz 모두 수집. 동일 patient_id가 양쪽에 있으면 .nii 우선.
    """
    nii_files = {_extract_patient_id(p): p for p in sorted(input_dir.glob("*.nii"))}
    for p in sorted(input_dir.glob("*.nii.gz")):
        pid = _extract_patient_id(p)
        if pid not in nii_files:
            nii_files[pid] = p
    return sorted(nii_files.values(), key=lambda p: _extract_patient_id(p))


def process_volume_masks(nifti_path, output_path):
    """
    단일 volume의 모든 slice에 대해 action masks 계산 후 .npz로 저장.
    """
    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata().astype(np.float32)

    shape = volume.shape
    z_axis = int(np.argmin(shape))
    n_slices = shape[z_axis]

    # 마스크 저장용 (Z, H, W) - uint8
    if z_axis == 2:
        H, W = shape[0], shape[1]
    elif z_axis == 0:
        H, W = shape[1], shape[2]
    else:
        H, W = shape[0], shape[2]

    artifact_vol = np.zeros((n_slices, H, W), dtype=np.uint8)
    fluid_vol = np.zeros((n_slices, H, W), dtype=np.uint8)
    structure_vol = np.zeros((n_slices, H, W), dtype=np.uint8)

    for z in range(n_slices):
        if z_axis == 0:
            image_hu = volume[z, :, :]
        elif z_axis == 1:
            image_hu = volume[:, z, :]
        else:
            image_hu = volume[:, :, z]

        try:
            masks, _ = generate_action_masks(image_hu, verbose=False)

            artifact_vol[z] = (np.clip(masks['artifact'], 0, 1) * 255).astype(np.uint8)
            fluid_vol[z] = (np.clip(masks['fluid'], 0, 1) * 255).astype(np.uint8)
            structure_vol[z] = (np.clip(masks['structure'], 0, 1) * 255).astype(np.uint8)
        except Exception as e:
            print(f"    Slice {z} error: {e}")

    np.savez_compressed(output_path,
                        artifact=artifact_vol,
                        fluid=fluid_vol,
                        structure=structure_vol)

    # 통계
    total_art = (artifact_vol > 76).sum()   # > 0.3 × 255
    total_flu = (fluid_vol > 76).sum()
    return n_slices, total_art, total_flu


def main():
    parser = argparse.ArgumentParser(description="Pre-compute action-guided masks")
    parser.add_argument('--input_dir', type=str,
                        default="F:/LD-CT SR/Data/NC-CT NIfTI",
                        help="NC-CT NIfTI root directory")
    parser.add_argument('--output_dir', type=str,
                        default="F:/LD-CT SR/Data/NC-CT Masks",
                        help="Output mask directory")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing mask files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = _collect_nifti_files(input_dir)

    print(f"{'='*60}")
    print(f"Pre-compute Action-Guided Masks")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(nifti_files)}")
    print(f"Force:  {args.force}")
    print(f"{'='*60}\n")

    t_start = time.time()
    total_files = len(nifti_files)
    processed = 0

    for i, nifti_path in enumerate(nifti_files):
        patient_id = _extract_patient_id(nifti_path)
        output_path = output_dir / f"{patient_id}.npz"

        if output_path.exists() and not args.force:
            print(f"[{i+1}/{total_files}] Skip (exists): {patient_id}")
            continue

        t1 = time.time()
        try:
            n_slices, n_art, n_flu = process_volume_masks(
                str(nifti_path), str(output_path)
            )
            elapsed = time.time() - t1
            processed += 1

            # ETA는 실제 처리된 파일 기준
            remaining = total_files - (i + 1)
            avg_time = (time.time() - t_start) / processed
            eta_min = remaining * avg_time / 60

            size_mb = output_path.stat().st_size / 1024 / 1024

            print(f"[{i+1}/{total_files}] {patient_id}: "
                  f"{n_slices}sl, art={n_art}, flu={n_flu}, "
                  f"{size_mb:.1f}MB, {elapsed:.0f}s "
                  f"(ETA: {eta_min:.0f}min)")

        except Exception as e:
            print(f"[{i+1}/{total_files}] ERROR {patient_id}: {e}")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! Processed: {processed}/{total_files}, Total: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
