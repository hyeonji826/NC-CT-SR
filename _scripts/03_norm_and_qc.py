import os
import sys
import csv
import math
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm  # ✅ 추가

# -----------------------
# Utils
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_image(p: Path) -> sitk.Image:
    if not p.exists():
        raise FileNotFoundError(str(p))
    return sitk.ReadImage(str(p))

def to_np(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)

def from_np(arr: np.ndarray, ref: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref)
    return out

def nearly_equal(a, b, tol=1e-6):
    return abs(a - b) <= tol

def voxel_radius_mm_to_pixels(spacing: Tuple[float, float, float], radius_mm: float) -> Tuple[int, int, int]:
    return tuple(max(1, int(round(radius_mm / s))) for s in spacing)

def resample_to_reference(moving: sitk.Image, reference: sitk.Image, default_value: float = -1024.0) -> sitk.Image:
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        default_value,
        moving.GetPixelID(),
    )

def check_same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and all(nearly_equal(sa, sb) for sa, sb in zip(a.GetSpacing(), b.GetSpacing()))
        and all(nearly_equal(oa, ob) for oa, ob in zip(a.GetOrigin(), b.GetOrigin()))
        and all(nearly_equal(da, db) for da, db in zip(a.GetDirection(), b.GetDirection()))
    )

def clip_window(arr: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    return np.clip(arr, wmin, wmax, out=np.empty_like(arr))

def normalize_0_1(arr: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    arr = (arr - wmin) / (wmax - wmin)
    return np.clip(arr, 0.0, 1.0, out=np.empty_like(arr))

def normalize_m1_1(arr: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    arr = 2.0 * (arr - wmin) / (wmax - wmin) - 1.0
    return np.clip(arr, -1.0, 1.0, out=np.empty_like(arr))

def make_body_mask(img_hu_np: np.ndarray, threshold_hu: float, spacing_xyz: Tuple[float, float, float], closing_mm=3.0) -> np.ndarray:
    mask = (img_hu_np > threshold_hu).astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.SetSpacing(spacing_xyz[::-1])
    rad_pix = voxel_radius_mm_to_pixels(spacing_xyz, closing_mm)
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    closing.SetKernelRadius(rad_pix[::-1])
    closing.SetForegroundValue(1)
    mask_img = closing.Execute(mask_img)
    fill = sitk.VotingBinaryHoleFillingImageFilter()
    fill.SetForegroundValue(1)
    fill.SetBackgroundValue(0)
    fill.SetRadius(rad_pix[::-1])
    fill.SetMajorityThreshold(1)
    mask_img = fill.Execute(mask_img)
    cc = sitk.ConnectedComponent(mask_img)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.BinaryThreshold(relabeled, 1, 1, 1, 0)
    return sitk.GetArrayFromImage(largest).astype(np.uint8)

def intensity_stats(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if mask is not None:
        m = mask.astype(bool)
        vals = arr[m] if np.any(m) else arr
        body_ratio = float(np.mean(m))
    else:
        vals = arr
        body_ratio = None
    if vals.size == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, std=np.nan, p1=np.nan, p99=np.nan, body_ratio=body_ratio)
    p1 = float(np.percentile(vals, 1))
    p99 = float(np.percentile(vals, 99))
    return dict(
        min=float(np.min(vals)),
        max=float(np.max(vals)),
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        p1=p1,
        p99=p99,
        body_ratio=body_ratio,
    )

def write_image_float32(img_np_norm: np.ndarray, ref: sitk.Image, out_path: Path):
    out_img = from_np(img_np_norm.astype(np.float32), ref)
    ensure_dir(out_path.parent)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(out_path))
    writer.UseCompressionOn()
    writer.Execute(out_img)

def write_yaml_key(yaml_path: Path, key: str, value: Any):
    try:
        import yaml
        ensure_dir(yaml_path.parent)
        data = {}
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                try:
                    data = yaml.safe_load(f) or {}
                except Exception:
                    data = {}
        data[key] = value
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    except Exception:
        pass

# -----------------------
# Core
# -----------------------
def process_case(id7: str, paths: Dict[str, Path], args: argparse.Namespace):
    note = ""
    try:
        nc_img = read_image(paths["nc_resampled"])
        ce_img = read_image(paths["ce_raw"])

        if not check_same_grid(nc_img, ce_img):
            if args.auto_fix_grid:
                nc_img = resample_to_reference(nc_img, ce_img, default_value=args.window_min)
                note += "Resampled grid; "
            else:
                raise RuntimeError("Grid mismatch")

        nc_np = to_np(nc_img).astype(np.float32)
        ce_np = to_np(ce_img).astype(np.float32)
        nc_clip = clip_window(nc_np, args.window_min, args.window_max)
        ce_clip = clip_window(ce_np, args.window_min, args.window_max)

        if args.masking:
            spacing = tuple(nc_img.GetSpacing()[::-1])
            mask = make_body_mask(nc_clip, args.mask_threshold, spacing, args.mask_closing_mm)
        else:
            mask = None

        if args.norm_mode == "zero_one":
            nc_norm = normalize_0_1(nc_clip, args.window_min, args.window_max)
            ce_norm = normalize_0_1(ce_clip, args.window_min, args.window_max)
        else:
            nc_norm = normalize_m1_1(nc_clip, args.window_min, args.window_max)
            ce_norm = normalize_m1_1(ce_clip, args.window_min, args.window_max)

        if mask is not None:
            bg = 0.0
            inv = (mask == 0)
            nc_norm[inv] = bg
            ce_norm[inv] = bg

        write_image_float32(nc_norm, ce_img, paths["nc_norm_out"])
        write_image_float32(ce_norm, ce_img, paths["ce_norm_out"])

        return {"status": "OK", "note": note}
    except Exception as e:
        return {"status": "FAIL", "note": f"{type(e).__name__}: {e}"}

def find_all_ids(root: Path) -> list:
    nc_root = root / "Data" / "nii_preproc" / "NC"
    ids = []
    for d in nc_root.glob("*"):
        if (d / "NC_resampled.nii.gz").exists():
            ids.append(d.name)
    return sorted(ids)

def build_paths(root: Path, id7: str) -> Dict[str, Path]:
    return {
        "nc_resampled": root / "Data" / "nii_preproc" / "NC" / id7 / "NC_resampled.nii.gz",
        "ce_raw": root / "Data" / "nii_raw" / "CE_D" / id7 / "CE_raw.nii.gz",
        "nc_norm_out": root / "Data" / "nii_preproc_norm" / "NC" / id7 / "NC_norm.nii.gz",
        "ce_norm_out": root / "Data" / "nii_preproc_norm" / "CE" / id7 / "CE_norm.nii.gz",
    }

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="LD-CT SR normalization (with tqdm)")
    parser.add_argument("--root", required=True)
    parser.add_argument("--window-min", type=float, default=-150)
    parser.add_argument("--window-max", type=float, default=250)
    parser.add_argument("--norm-mode", choices=["zero_one", "minus1_1"], default="zero_one")
    parser.add_argument("--masking", action="store_true", default=True)
    parser.add_argument("--mask-threshold", type=float, default=-200)
    parser.add_argument("--mask-closing-mm", type=float, default=3.0)
    parser.add_argument("--auto-fix-grid", action="store_true", default=True)
    args = parser.parse_args()

    root = Path(args.root)
    ids = find_all_ids(root)

    reports_dir = root / "Outputs" / "reports"
    ensure_dir(reports_dir)
    log_path = reports_dir / "preprocessing_log.csv"

    log_rows = []

    print(f"[INFO] Total cases: {len(ids)}")
    for id7 in tqdm(ids, desc="Normalizing volumes", ncols=80):
        paths = build_paths(root, id7)
        res = process_case(id7, paths, args)
        log_rows.append({"id7": id7, **res})

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"[DONE] Log saved to: {log_path}")

if __name__ == "__main__":
    main()