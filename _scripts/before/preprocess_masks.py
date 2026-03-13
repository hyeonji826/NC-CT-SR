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
"""

import numpy as np
import nibabel as nib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_guided_masks import generate_action_masks


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
            masks, _ = generate_action_masks(image_hu)

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
    input_dir = Path("F:/LD-CT SR/Data/NC-CT NIfTI")
    output_dir = Path("F:/LD-CT SR/Data/NC-CT Masks")
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.glob("*.nii"))
    if not nifti_files:
        nifti_files = sorted(input_dir.glob("*.nii.gz"))

    print(f"{'='*60}")
    print(f"Pre-compute Action-Guided Masks")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(nifti_files)}")
    print(f"{'='*60}\n")

    t_start = time.time()
    total_files = len(nifti_files)

    for i, nifti_path in enumerate(nifti_files):
        patient_id = nifti_path.stem
        output_path = output_dir / f"{patient_id}.npz"

        if output_path.exists():
            print(f"[{i+1}/{total_files}] Skip (exists): {patient_id}")
            continue

        t1 = time.time()
        try:
            n_slices, n_art, n_flu = process_volume_masks(
                str(nifti_path), str(output_path)
            )
            elapsed = time.time() - t1

            done = i + 1
            remaining = total_files - done
            avg_time = (time.time() - t_start) / done
            eta_min = remaining * avg_time / 60

            size_mb = output_path.stat().st_size / 1024 / 1024

            print(f"[{done}/{total_files}] {patient_id}: "
                  f"{n_slices}sl, art={n_art}, flu={n_flu}, "
                  f"{size_mb:.1f}MB, {elapsed:.0f}s "
                  f"(ETA: {eta_min:.0f}min)")

        except Exception as e:
            print(f"[{i+1}/{total_files}] ERROR {patient_id}: {e}")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! Total: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
