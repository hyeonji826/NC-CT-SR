"""
NC-CT Original vs Processed 비교 이미지 생성.
Original | Processed | Diff(x10)
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def hu_to_display(hu_array, wl=40, ww=400):
    """HU → [0,255] with Window Level/Width"""
    hu_min = wl - ww / 2
    hu_max = wl + ww / 2
    clipped = np.clip(hu_array, hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)

def orient(arr):
    """Raw → Display 회전보정"""
    return np.flipud(np.rot90(arr, k=1))

def get_slice(volume, z_axis, z):
    if z_axis == 0:
        return volume[z, :, :]
    elif z_axis == 1:
        return volume[:, z, :]
    else:
        return volume[:, :, z]

def main():
    original_dir = Path("F:/LD-CT SR/Data/NC-CT NIfTI")
    processed_dir = Path("F:/LD-CT SR/Data/NC-CT Processed")
    output_dir = Path("F:/LD-CT SR/Outputs/processed_preview")
    output_dir.mkdir(parents=True, exist_ok=True)

    slices = [
        ("7251764", 40),
        ("0452240", 33),
        ("0090775", 44),
        ("1228288", 30),
        ("1116178", 30),
    ]

    for patient_id, z in slices:
        orig_path = original_dir / f"{patient_id}.nii"
        proc_path = processed_dir / f"{patient_id}.nii"

        if not orig_path.exists():
            print(f"[SKIP] Original not found: {orig_path}")
            continue
        if not proc_path.exists():
            print(f"[SKIP] Processed not found: {proc_path}")
            continue

        print(f"\n{patient_id} slice {z}:")

        orig_vol = nib.load(str(orig_path)).get_fdata().astype(np.float32)
        proc_vol = nib.load(str(proc_path)).get_fdata().astype(np.float32)

        z_axis = int(np.argmin(orig_vol.shape))

        orig_hu = get_slice(orig_vol, z_axis, z)
        proc_hu = get_slice(proc_vol, z_axis, z)

        # 차이 통계
        diff = proc_hu - orig_hu
        changed = np.abs(diff) > 0.5
        n_changed = changed.sum()
        print(f"  Changed pixels: {n_changed} ({n_changed/orig_hu.size*100:.2f}%)")
        if n_changed > 0:
            print(f"  Mean |diff|: {np.mean(np.abs(diff[changed])):.1f} HU")
            print(f"  Max  |diff|: {np.max(np.abs(diff[changed])):.1f} HU")

        # Display
        orig_disp = orient(hu_to_display(orig_hu))
        proc_disp = orient(hu_to_display(proc_hu))

        # Diff map (x10 amplification)
        diff_disp = orient(diff)
        diff_amp = np.clip(np.abs(diff_disp) * 10, 0, 255).astype(np.uint8)

        # 3-panel: Original | Processed | Diff(x10)
        h, w = orig_disp.shape
        gap = 4
        canvas_w = w * 3 + gap * 2
        header_h = 30
        canvas = Image.new('L', (canvas_w, h + header_h), 0)

        canvas.paste(Image.fromarray(orig_disp), (0, header_h))
        canvas.paste(Image.fromarray(proc_disp), (w + gap, header_h))
        canvas.paste(Image.fromarray(diff_amp), (w * 2 + gap * 2, header_h))

        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        draw.text((w // 2 - 30, 5), "Original", fill=255, font=font)
        draw.text((w + gap + w // 2 - 40, 5), "Processed", fill=255, font=font)
        draw.text((w * 2 + gap * 2 + w // 2 - 40, 5), "Diff x10", fill=255, font=font)

        out_path = output_dir / f"{patient_id}_s{z}.png"
        canvas.save(str(out_path))
        print(f"  Saved: {out_path.name}")

    print(f"\nDone! Output: {output_dir}")

if __name__ == "__main__":
    main()
