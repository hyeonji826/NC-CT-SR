"""
3D Streak Detection — Test & Visualization
============================================
환자 1명에 대해 streak detection 실행 후
axial / coronal / sagittal 3-view 시각화 (탐지만, 복원 X)
"""

import os
import sys
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 같은 디렉토리의 모듈
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streak_removal_3d import load_volume, detect_streaks_3d


# ============================================================
# Display helpers
# ============================================================

def orient_axial(arr):
    """NC-CT NIfTI axial display: rot90(-1) + fliplr"""
    return np.fliplr(np.rot90(arr, k=-1))


def hu_to_display(arr, wl=-10, ww=400):
    """HU → 0~1 grayscale (window level / width)"""
    lo = wl - ww / 2
    hi = wl + ww / 2
    return np.clip((arr - lo) / (hi - lo), 0, 1)


# ============================================================
# Visualization
# ============================================================

def plot_axial(volume, anatomy_z, streak_map, z_indices, out_dir, patient_id):
    """Axial view: original / anatomy (z-median) / streak map"""
    n = len(z_indices)
    fig, axes = plt.subplots(3, n, figsize=(6 * n, 17))
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, z in enumerate(z_indices):
        orig = hu_to_display(orient_axial(volume[:, :, z]))
        anat = hu_to_display(orient_axial(anatomy_z[:, :, z]))
        stk = orient_axial(streak_map[:, :, z])

        axes[0, col].imshow(orig, cmap='gray')
        axes[0, col].set_title(f'Original (z={z})', fontsize=11)
        axes[0, col].axis('off')

        axes[1, col].imshow(anat, cmap='gray')
        axes[1, col].set_title(f'Anatomy z-median (z={z})', fontsize=11)
        axes[1, col].axis('off')

        vmax = max(np.abs(stk).max(), 5)
        axes[2, col].imshow(stk, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[2, col].set_title(f'Streak [{stk.min():.0f},{stk.max():.0f}] HU', fontsize=11)
        axes[2, col].axis('off')

    fig.suptitle(f'Axial — {patient_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'axial_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_coronal(volume, anatomy_z, streak_map, y_idx, out_dir, patient_id):
    """Coronal view (xz plane): streak이 수평 줄무늬로 보여야 함"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    orig = hu_to_display(volume[y_idx, :, :].T)
    anat = hu_to_display(anatomy_z[y_idx, :, :].T)
    diff = streak_map[y_idx, :, :].T

    axes[0].imshow(orig, cmap='gray', aspect='auto')
    axes[0].set_title('Original', fontsize=12)
    axes[0].set_ylabel('Z (slice)')
    axes[0].set_xlabel('X')

    axes[1].imshow(anat, cmap='gray', aspect='auto')
    axes[1].set_title('Anatomy (z-median)', fontsize=12)
    axes[1].set_xlabel('X')

    vmax = max(np.abs(diff).max(), 5)
    im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[2].set_title('Streak Map', fontsize=12)
    axes[2].set_xlabel('X')
    plt.colorbar(im, ax=axes[2], fraction=0.046, label='HU')

    fig.suptitle(f'Coronal (y={y_idx}) — {patient_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'coronal_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_sagittal(volume, anatomy_z, streak_map, x_idx, out_dir, patient_id):
    """Sagittal view (yz plane)"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    orig = hu_to_display(volume[:, x_idx, :].T)
    anat = hu_to_display(anatomy_z[:, x_idx, :].T)
    diff = streak_map[:, x_idx, :].T

    axes[0].imshow(orig, cmap='gray', aspect='auto')
    axes[0].set_title('Original', fontsize=12)
    axes[0].set_ylabel('Z (slice)')
    axes[0].set_xlabel('Y')

    axes[1].imshow(anat, cmap='gray', aspect='auto')
    axes[1].set_title('Anatomy (z-median)', fontsize=12)
    axes[1].set_xlabel('Y')

    vmax = max(np.abs(diff).max(), 5)
    im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[2].set_title('Streak Map', fontsize=12)
    axes[2].set_xlabel('Y')
    plt.colorbar(im, ax=axes[2], fraction=0.046, label='HU')

    fig.suptitle(f'Sagittal (x={x_idx}) — {patient_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'sagittal_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_streak_overlay(volume, streak_est, z_idx, out_dir, patient_id):
    """Axial streak overlay: 원본 위에 streak을 red/blue로 표시"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    orig_disp = hu_to_display(orient_axial(volume[:, :, z_idx]))
    stk = orient_axial(streak_est[:, :, z_idx])

    # Original
    axes[0].imshow(orig_disp, cmap='gray')
    axes[0].set_title('Original', fontsize=12)
    axes[0].axis('off')

    # Streak field
    vmax = max(np.abs(stk).max(), 5)
    im = axes[1].imshow(stk, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[1].set_title(f'Streak Field [{stk.min():.1f}, {stk.max():.1f}] HU', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # Overlay: red = positive (bright artifact), blue = negative (dark artifact)
    rgb = np.stack([orig_disp] * 3, axis=-1).copy()
    stk_norm = np.clip(stk / max(vmax, 1), -1, 1)
    # Positive streak → add red
    pos = stk_norm > 0.05
    rgb[pos, 0] = np.clip(rgb[pos, 0] + stk_norm[pos] * 0.5, 0, 1)
    rgb[pos, 1] = np.clip(rgb[pos, 1] - stk_norm[pos] * 0.3, 0, 1)
    rgb[pos, 2] = np.clip(rgb[pos, 2] - stk_norm[pos] * 0.3, 0, 1)
    # Negative streak → add blue
    neg = stk_norm < -0.05
    rgb[neg, 2] = np.clip(rgb[neg, 2] - stk_norm[neg] * 0.5, 0, 1)
    rgb[neg, 0] = np.clip(rgb[neg, 0] + stk_norm[neg] * 0.3, 0, 1)
    rgb[neg, 1] = np.clip(rgb[neg, 1] + stk_norm[neg] * 0.3, 0, 1)

    axes[2].imshow(rgb)
    axes[2].set_title('Overlay (Red=bright, Blue=dark streak)', fontsize=12)
    axes[2].axis('off')

    fig.suptitle(f'Streak Analysis (z={z_idx}) — {patient_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'streak_overlay.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='3D Streak Detection Test')
    parser.add_argument('--patient', default=None, help='Patient ID (e.g. 0025980)')
    parser.add_argument('--nifti_dir', default='F:/LD-CT SR/Data/NC-CT NIfTI')
    parser.add_argument('--output_dir', default='F:/LD-CT SR/Outputs/streak_removal_3d')
    parser.add_argument('--dcm_dir', default='F:/LD-CT SR/Data/HCC Abd NC-CT')
    parser.add_argument('--kernel_z', type=int, default=5)
    args = parser.parse_args()

    # Find patient NIfTI
    nifti_files = sorted([f for f in os.listdir(args.nifti_dir)
                          if f.endswith(('.nii', '.nii.gz'))])
    print(f"Available: {len(nifti_files)} patients")

    if args.patient:
        match = [f for f in nifti_files if args.patient in f]
        if not match:
            print(f"Patient '{args.patient}' not found!")
            return
        nifti_name = match[0]
    else:
        nifti_name = nifti_files[0]

    nifti_path = os.path.join(args.nifti_dir, nifti_name)
    patient_id = nifti_name.replace('.nii.gz', '').replace('.nii', '')
    print(f"Patient: {patient_id}")
    print(f"File: {nifti_path}")

    # Read DICOM metadata (exposure mAs)
    exposure_mAs = None
    dcm_patient_dir = os.path.join(args.dcm_dir, patient_id)
    if os.path.isdir(dcm_patient_dir):
        try:
            import pydicom
            dcm_files = sorted([f for f in os.listdir(dcm_patient_dir) if f.endswith('.dcm')])
            if dcm_files:
                ds = pydicom.dcmread(os.path.join(dcm_patient_dir, dcm_files[0]),
                                     stop_before_pixels=True)
                exposure_mAs = int(ds[0x0018, 0x1152].value)
                kvp = float(ds[0x0018, 0x0060].value)
                print(f"DICOM: kVp={kvp:.0f}, Exposure={exposure_mAs} mAs")
        except Exception as e:
            print(f"DICOM read failed: {e}")
    else:
        print(f"No DICOM dir for {patient_id}, using default noise_factor=1.0")

    # Load
    print("Loading volume...")
    volume, affine, header = load_volume(nifti_path)
    H, W, Z = volume.shape
    print(f"Volume: {H}x{W}x{Z}, HU: [{volume.min():.0f}, {volume.max():.0f}]")

    # Run streak detection
    streak_map, anatomy_z, body_3d = detect_streaks_3d(
        volume,
        kernel_z=args.kernel_z,
        exposure_mAs=exposure_mAs,
    )

    # Output directory
    out_dir = os.path.join(args.output_dir, patient_id)
    os.makedirs(out_dir, exist_ok=True)

    # Representative slice indices
    z_mid = Z // 2
    z_indices = [max(0, z_mid - 10), z_mid, min(Z - 1, z_mid + 10)]
    y_mid = H // 2
    x_mid = W // 2

    print(f"\nGenerating visualizations...")

    # Axial: original / anatomy / streak (3 slices)
    plot_axial(volume, anatomy_z, streak_map, z_indices, out_dir, patient_id)

    # Coronal
    plot_coronal(volume, anatomy_z, streak_map, y_mid, out_dir, patient_id)

    # Sagittal
    plot_sagittal(volume, anatomy_z, streak_map, x_mid, out_dir, patient_id)

    # Streak overlay (middle slice)
    plot_streak_overlay(volume, streak_map, z_mid, out_dir, patient_id)

    # Print summary stats
    body_stk = streak_map[body_3d]
    print(f"\n=== Summary (whole volume, body only) ===")
    print(f"  Streak: mean={body_stk.mean():.2f}, "
          f"std={body_stk.std():.2f}, "
          f"max|s|={np.abs(body_stk).max():.1f} HU")
    print(f"  |streak| > 5 HU:  {(np.abs(body_stk) > 5).sum()} voxels")
    print(f"  |streak| > 10 HU: {(np.abs(body_stk) > 10).sum()} voxels")
    print(f"  |streak| > 30 HU: {(np.abs(body_stk) > 30).sum()} voxels")
    print(f"\nAll outputs -> {out_dir}")


if __name__ == '__main__':
    main()
