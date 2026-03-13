"""
v3 마스크 테스트: 마스크 + photon starvation 감지 + 수직보간 결과 시각화.
기존 v2와 비교 가능하도록 동일 환자/슬라이스 사용.
"""

import numpy as np
import nibabel as nib
import os
import sys
import io
from pathlib import Path
from contextlib import redirect_stdout

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_guided_masks import (
    find_body_mask, detect_arm_artifact_band, detect_photon_starvation,
)
from action_guided_masks import generate_action_masks as generate
from preprocess_targets import inpaint_arm_streaks_quiet
from dataset import load_nifti_volume_autoaxis

TEST_CASES = [
    ("1728852", [0.15, 0.35, 0.55, 0.75]),
    ("0025980", [0.15, 0.35, 0.55, 0.75]),
    ("0040386", [0.15, 0.35, 0.55, 0.75]),
    ("1192127", [0.15, 0.35, 0.55, 0.75]),
    ("7032506", [0.15, 0.35, 0.55, 0.75]),
]

OUTPUT_DIR = Path("F:/LD-CT SR/Outputs/mask_test_v3")
NIFTI_ROOT = Path("F:/LD-CT SR/Data/NC-CT NIfTI")


def orient(arr):
    return np.flipud(np.rot90(arr, k=1))


def run_quiet(func, *args, **kwargs):
    f = io.StringIO()
    with redirect_stdout(f):
        return func(*args, **kwargs)


def streak_correct(image_hu):
    """기존 가로 streak inpainting만 적용. Starvation 감지는 별도."""
    body_mask = find_body_mask(image_hu)
    result = detect_arm_artifact_band(image_hu, body_mask, margin=30)
    arm_band = result[0] if isinstance(result, tuple) else result

    n_streak = 0
    corrected = image_hu.copy()
    starv_mask = np.zeros_like(image_hu, dtype=bool)

    if arm_band.any():
        # 가로 streak inpainting만 (기존 방식)
        corrected, n_streak = inpaint_arm_streaks_quiet(corrected, arm_band, body_mask)

        # Starvation 감지 (시각화 + artifact mask용, inpainting 안함)
        starv_mask, _ = detect_photon_starvation(image_hu, body_mask, arm_band)

    return corrected, n_streak, starv_mask


def visualize_patient(patient_id, slice_ratios, output_dir):
    nifti_path = NIFTI_ROOT / f"{patient_id}.nii"
    if not nifti_path.exists():
        nifti_path = NIFTI_ROOT / f"{patient_id}.nii.gz"
    if not nifti_path.exists():
        print(f"  [SKIP] {patient_id}")
        return

    vol = load_nifti_volume_autoaxis(str(nifti_path))
    n_slices = vol.shape[0]
    slice_indices = [min(int(n_slices * r), n_slices - 1) for r in slice_ratios]

    n_rows = len(slice_indices)
    # 6열: CT | v3 Artifact | Starvation | v3 Fluid | Corrected | Action Map
    fig, axes = plt.subplots(n_rows, 6, figsize=(30, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, sidx in enumerate(slice_indices):
        image_hu = vol[sidx].astype(np.float32)

        # v3 마스크
        masks_v3, feats_v3 = run_quiet(generate, image_hu)

        # Streak 보정 (가로 inpainting만) + starvation 감지
        corrected, n_streak, starv_mask = streak_correct(image_hu)

        # CT display
        hu_disp = np.clip((image_hu + 160) / 400, 0, 1)
        ct_uint8 = (hu_disp * 255).astype(np.uint8)
        ct_rgb = np.stack([ct_uint8]*3, axis=-1).astype(np.float32)

        # 보정 결과 display
        corr_disp = np.clip((corrected + 160) / 400, 0, 1)
        corr_uint8 = (corr_disp * 255).astype(np.uint8)

        # 1. Original CT
        axes[row, 0].imshow(orient(ct_uint8), cmap='gray')
        axes[row, 0].set_title(f'Slice {sidx}/{n_slices} ({slice_ratios[row]*100:.0f}%)', fontsize=11)
        axes[row, 0].axis('off')

        # 2. v3 Artifact (red overlay, starvation 포함)
        art_color = np.zeros_like(ct_rgb); art_color[:, :, 0] = 255
        overlay_v3 = ct_rgb.copy()
        alpha3 = masks_v3['artifact'][:, :, np.newaxis] * 0.6
        overlay_v3 = overlay_v3 * (1 - alpha3) + art_color * alpha3
        v3_count = (masks_v3['artifact'] > 0.3).sum()
        axes[row, 1].imshow(orient(overlay_v3.astype(np.uint8)))
        axes[row, 1].set_title(f'v3 Artifact ({v3_count}px)', fontsize=10)
        axes[row, 1].axis('off')

        # 3. Photon Starvation (yellow overlay)
        overlay_starv = ct_rgb.copy()
        starv_color = np.zeros_like(ct_rgb)
        starv_color[:, :, 0] = 255; starv_color[:, :, 1] = 255
        starv_alpha = starv_mask[:, :, np.newaxis].astype(np.float32) * 0.5
        overlay_starv = overlay_starv * (1 - starv_alpha) + starv_color * starv_alpha
        starv_count = starv_mask.sum()
        axes[row, 2].imshow(orient(overlay_starv.astype(np.uint8)))
        axes[row, 2].set_title(f'Starvation ({starv_count}px)', fontsize=10)
        axes[row, 2].axis('off')

        # 4. v3 Fluid (blue overlay)
        overlay_flu = ct_rgb.copy()
        flu_color = np.zeros_like(ct_rgb); flu_color[:, :, 2] = 255
        alpha_f = masks_v3['fluid'][:, :, np.newaxis] * 0.6
        overlay_flu = overlay_flu * (1 - alpha_f) + flu_color * alpha_f
        flu_count = (masks_v3['fluid'] > 0.3).sum()
        axes[row, 3].imshow(orient(overlay_flu.astype(np.uint8)))
        axes[row, 3].set_title(f'v3 Fluid ({flu_count}px)', fontsize=10)
        axes[row, 3].axis('off')

        # 5. Corrected CT (가로 streak inpainting만)
        axes[row, 4].imshow(orient(corr_uint8), cmap='gray')
        axes[row, 4].set_title(f'Corrected ({n_streak}px)', fontsize=10)
        axes[row, 4].axis('off')

        # 6. Action Map (R=artifact, G=structure, B=fluid)
        H, W = image_hu.shape
        action_map = np.zeros((H, W, 3), dtype=np.uint8)
        action_map[:, :, 1] = (masks_v3['structure'] * 200).astype(np.uint8)
        art_s = masks_v3['artifact'] > 0.15
        action_map[art_s, 0] = (masks_v3['artifact'][art_s] * 255).astype(np.uint8)
        action_map[art_s, 1] = 0
        flu_s = masks_v3['fluid'] > 0.2
        action_map[flu_s, 2] = (masks_v3['fluid'][flu_s] * 255).astype(np.uint8)
        axes[row, 5].imshow(orient(action_map))
        axes[row, 5].set_title('Action Map', fontsize=10)
        axes[row, 5].axis('off')

    plt.suptitle(f'Patient {patient_id} — v3 Mask + Starvation Detection',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    out_path = output_dir / f'v3_test_{patient_id}.png'
    plt.savefig(str(out_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("v3 Mask Test + Photon Starvation Correction")
    print("=" * 60)

    for patient_id, ratios in TEST_CASES:
        print(f"\n[{patient_id}]")
        visualize_patient(patient_id, ratios, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"All results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
