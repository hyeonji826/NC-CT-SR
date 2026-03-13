"""
간단 마스크 테스트: Original vs Corrected 비교 (환자별 1슬라이스)
"""

import numpy as np
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from action_guided_masks import find_body_mask, detect_arm_artifact_band
from preprocess_targets import inpaint_arm_streaks_quiet
from dataset import load_nifti_volume_autoaxis

TEST_CASES = [
    ("1728852", 0.35),
    ("0025980", 0.35),
    ("0040386", 0.35),
    ("1192127", 0.35),
    ("7032506", 0.35),
]

OUTPUT_DIR = Path("F:/LD-CT SR/Outputs/ma_hybrid/integrated_test")
NIFTI_ROOT = Path("F:/LD-CT SR/Data/NC-CT NIfTI")


def orient(arr):
    return np.flipud(np.rot90(arr, k=1))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n = len(TEST_CASES)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))

    for row, (pid, ratio) in enumerate(TEST_CASES):
        nifti_path = NIFTI_ROOT / f"{pid}.nii"
        if not nifti_path.exists():
            nifti_path = NIFTI_ROOT / f"{pid}.nii.gz"
        if not nifti_path.exists():
            print(f"  [SKIP] {pid}")
            continue

        vol = load_nifti_volume_autoaxis(str(nifti_path))
        n_slices = vol.shape[0]
        sidx = min(int(n_slices * ratio), n_slices - 1)
        image_hu = vol[sidx].astype(np.float32)

        # Streak correction
        body_mask = find_body_mask(image_hu)
        result = detect_arm_artifact_band(image_hu, body_mask, margin=30)
        arm_band = result[0] if isinstance(result, tuple) else result
        corrected = image_hu.copy()
        n_streak = 0
        if arm_band.any():
            corrected, n_streak = inpaint_arm_streaks_quiet(
                corrected, arm_band, body_mask
            )

        # Display
        orig = (np.clip((image_hu + 160) / 400, 0, 1) * 255).astype(np.uint8)
        corr = (np.clip((corrected + 160) / 400, 0, 1) * 255).astype(np.uint8)

        axes[row, 0].imshow(orient(orig), cmap='gray')
        axes[row, 0].set_title(f'{pid}  Slice {sidx}/{n_slices}', fontsize=11)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(orient(corr), cmap='gray')
        axes[row, 1].set_title(f'Corrected ({n_streak}px)', fontsize=11)
        axes[row, 1].axis('off')

    plt.suptitle('Original vs Corrected', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = OUTPUT_DIR / 'slice_vs_corrected.png'
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
