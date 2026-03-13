"""
Pre-compute streak-corrected NC-CT targets for training.
==========================================================

목적:
    Arm-to-arm streak artifact를 inpaint한 pseudo-clean target 생성.
    모델이 N2N으로 노이즈 제거를 학습하는 동시에,
    streak artifact 제거도 학습할 수 있도록 target에서 streak을 미리 보정.

처리:
    1. Body mask 검출
    2. Arm artifact band 검출
    3. Band 내 streak pixel만 주변 조직값으로 inpaint
    4. NIfTI로 저장 (원본 header/affine 보존)

일반 디노이징은 하지 않음 → 모델이 자체적으로 학습.
"""

import numpy as np
import nibabel as nib
import os
import sys
import time
from pathlib import Path
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_guided_masks import find_body_mask, detect_arm_artifact_band


def inpaint_arm_streaks_quiet(image_hu, arm_band, body_mask, radius=5):
    """
    Arm band 내 streak artifact 픽셀만 주변 조직값으로 inpaint.
    (print 없는 버전 - batch processing용)

    Returns:
        corrected: inpaint된 이미지 (float32)
        n_streak: 보정된 streak 픽셀 수
    """
    if not arm_band.any():
        return image_hu.copy(), 0

    output = image_hu.copy().astype(np.float64)

    # Body mask가 bool이 아닐 수 있음
    if body_mask.dtype != bool:
        body_bool = body_mask > 0.5
    else:
        body_bool = body_mask

    band_pixels = arm_band & body_bool

    # Band 내 non-water 참조값
    non_water_in_band = band_pixels & ((image_hu < -30) | (image_hu > 30))
    if non_water_in_band.any():
        ref_hu = np.median(image_hu[non_water_in_band])
    else:
        ref_hu = 40.0

    # Streak mask: band 내 water-like HU + 주변과 편차 큰 것
    streak_mask = band_pixels & (image_hu >= -30) & (image_hu < 30)
    local_mean = ndimage.uniform_filter(image_hu.astype(np.float64), size=7)
    hu_deviation = np.abs(image_hu - local_mean)
    streak_mask = streak_mask & (hu_deviation > 10)

    n_streak = int(streak_mask.sum())
    if n_streak == 0:
        return output.astype(np.float32), 0

    # Iterative boundary inpainting
    inpaint_result = output.copy()
    remaining = streak_mask.copy()

    for _ in range(radius * 2):
        if not remaining.any():
            break

        dilated = ndimage.binary_dilation(~remaining, iterations=1)
        border = remaining & dilated

        if not border.any():
            border = remaining

        valid = (~remaining).astype(np.float64)
        value_sum = ndimage.uniform_filter(inpaint_result * valid, size=3) * 9
        count = ndimage.uniform_filter(valid, size=3) * 9

        update_mask = border & (count > 0.5)
        if update_mask.any():
            inpaint_result[update_mask] = value_sum[update_mask] / count[update_mask]

        remaining[update_mask] = False

    if remaining.any():
        inpaint_result[remaining] = ref_hu

    output[streak_mask] = inpaint_result[streak_mask]
    return output.astype(np.float32), n_streak


def process_volume(input_path, output_path):
    """
    단일 volume 처리.
    NIfTI raw format: (H, W, Z) → volume[:, :, z]로 slice 접근.
    """
    nifti_img = nib.load(input_path)
    volume = nifti_img.get_fdata().astype(np.float32)

    # Shape 확인 (Z축 = 가장 작은 차원)
    shape = volume.shape
    z_axis = int(np.argmin(shape))

    if z_axis == 0:
        n_slices = shape[0]
    elif z_axis == 1:
        n_slices = shape[1]
    else:
        n_slices = shape[2]

    processed = volume.copy()
    total_streaks = 0
    slices_with_streaks = 0

    for z in range(n_slices):
        # Z축에 따라 slice 추출
        if z_axis == 0:
            image_hu = volume[z, :, :]
        elif z_axis == 1:
            image_hu = volume[:, z, :]
        else:
            image_hu = volume[:, :, z]

        # Body mask
        body_mask = find_body_mask(image_hu)
        if body_mask.sum() < 100:
            continue

        # Arm band detection
        arm_band, _ = detect_arm_artifact_band(image_hu, body_mask, margin=20)

        if arm_band.any():
            corrected, n_streak = inpaint_arm_streaks_quiet(
                image_hu, arm_band, body_mask
            )
            if n_streak > 0:
                # Write back
                if z_axis == 0:
                    processed[z, :, :] = corrected
                elif z_axis == 1:
                    processed[:, z, :] = corrected
                else:
                    processed[:, :, z] = corrected
                total_streaks += n_streak
                slices_with_streaks += 1

    # Save (preserve original NIfTI header and affine)
    out_img = nib.Nifti1Image(processed, nifti_img.affine, nifti_img.header)
    nib.save(out_img, output_path)

    return n_slices, total_streaks, slices_with_streaks


def main():
    input_dir = Path("F:/LD-CT SR/Data/NC-CT NIfTI")
    output_dir = Path("F:/LD-CT SR/Data/NC-CT Processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = sorted(input_dir.glob("*.nii"))
    if not nifti_files:
        nifti_files = sorted(input_dir.glob("*.nii.gz"))

    print(f"{'='*60}")
    print(f"Pre-compute Streak-Corrected Targets")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(nifti_files)}")
    print(f"{'='*60}")

    t_start = time.time()
    total_files = len(nifti_files)

    for i, nifti_path in enumerate(nifti_files):
        output_path = output_dir / nifti_path.name

        if output_path.exists():
            print(f"[{i+1}/{total_files}] Skip (exists): {nifti_path.name}")
            continue

        t1 = time.time()
        try:
            n_slices, n_streaks, n_slices_fixed = process_volume(
                str(nifti_path), str(output_path)
            )
            elapsed = time.time() - t1

            # ETA 계산
            done = i + 1
            remaining = total_files - done
            avg_time = (time.time() - t_start) / done
            eta_min = remaining * avg_time / 60

            print(f"[{done}/{total_files}] {nifti_path.name}: "
                  f"{n_slices} slices, {n_slices_fixed} fixed, "
                  f"{n_streaks} streak px, {elapsed:.1f}s "
                  f"(ETA: {eta_min:.0f}min)")

        except Exception as e:
            print(f"[{i+1}/{total_files}] ERROR {nifti_path.name}: {e}")
            # Error 시 원본 복사
            import shutil
            shutil.copy2(str(nifti_path), str(output_path))

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! Total: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
