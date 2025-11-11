import json
from pathlib import Path
from typing import Tuple
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import glob as pyglob

def load_config(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("mode", "index")
    cfg.setdefault("from_center", True)
    cfg.setdefault("center_xyz_index", [0, 0, 0])
    cfg.setdefault("size_xyz", [384, 384, 256])
    cfg.setdefault("pad_if_needed", False)
    cfg.setdefault("save_name", "NC_crop.nii")
    cfg.setdefault("center_fraction_xyz", None)
    cfg.setdefault("offset_xyz_index", [0,0,0])
    cfg.setdefault("offset_xyz_mm", [0,0,0])
    return cfg

def clamp(a, lo, hi): 
    return max(lo, min(hi, a))

def index_box_from_center(size_xyz_img, center_xyz_idx, box_size_xyz):
    sx, sy, sz = size_xyz_img
    cx, cy, cz = center_xyz_idx
    bx, by, bz = box_size_xyz
    hx, hy, hz = bx // 2, by // 2, bz // 2
    x0, y0, z0 = cx - hx, cy - hy, cz - hz
    x1, y1, z1 = x0 + bx, y0 + by, z0 + bz
    x0c, y0c, z0c = clamp(x0, 0, sx), clamp(y0, 0, sy), clamp(z0, 0, sz)
    x1c, y1c, z1c = clamp(x1, 0, sx), clamp(y1, 0, sy), clamp(z1, 0, sz)
    outx, outy, outz = max(0, x1c - x0c), max(0, y1c - y0c), max(0, z1c - z0c)
    return (x0c, y0c, z0c), (outx, outy, outz)

def crop_img(img: sitk.Image, start_xyz, size_xyz):
    if 0 in size_xyz:
        raise RuntimeError("Computed crop size has a zero dimension.")
    return sitk.RegionOfInterest(img, size_xyz, start_xyz)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=r"E:\LD-CT SR\configs\crop.json", type=str)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    paths = sorted(pyglob.glob(cfg["glob"]))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not paths:
        print(f"[ERR] No files matched: {cfg['glob']}")
        return

    print(f"[INFO] Files to crop: {len(paths)}")

    for p in tqdm(paths, desc="Cropping", ncols=100):
        p = Path(p)
        img = sitk.ReadImage(str(p))

        # --- 중심 계산 ---
        sx, sy, sz = img.GetSize()
        if cfg.get("center_fraction_xyz"):
            fx, fy, fz = cfg["center_fraction_xyz"]
            # fraction은 0~1 사이로 클램프
            fx = min(max(float(fx), 0.0), 1.0)
            fy = min(max(float(fy), 0.0), 1.0)
            fz = min(max(float(fz), 0.0), 1.0)
            center_idx = (int(round(sx*fx)), int(round(sy*fy)), int(round(sz*fz)))
        elif cfg["from_center"]:
            center_idx = (sx//2, sy//2, sz//2)
        else:
            center_idx = tuple(int(v) for v in cfg["center_xyz_index"])

        # index 오프셋
        if cfg["mode"] == "index":
            ox, oy, oz = (int(v) for v in cfg["offset_xyz_index"])
            center_idx = (center_idx[0]+ox, center_idx[1]+oy, center_idx[2]+oz)

        # physical(mm) 오프셋 (mode가 physical_mm일 때만)
        if cfg["mode"] == "physical_mm" and any(cfg["offset_xyz_mm"]):
            cx_mm = list(img.TransformIndexToPhysicalPoint(center_idx))
            cx_mm = [cx_mm[i] + float(cfg["offset_xyz_mm"][i]) for i in range(3)]
            center_idx = img.TransformPhysicalPointToIndex(tuple(cx_mm))

        # --- 크롭 계산/실행 ---
        start, size = index_box_from_center(
            img.GetSize(),
            center_idx,
            tuple(int(v) for v in cfg["size_xyz"])
        )
        cropped = crop_img(img, start, size)

        # 저장 경로: <output_dir>/<pid>/NC_crop.nii
        pid_dir = out_dir / p.parent.name
        pid_dir.mkdir(parents=True, exist_ok=True)
        out_path = pid_dir / cfg["save_name"]

        if args.dry_run:
            print(f"[DRY] {p.name} -> start={start}, size={size} -> {out_path}")
        else:
            sitk.WriteImage(cropped, str(out_path), useCompression=False)

        # --- QC (파일마다) ---
        arr = sitk.GetArrayFromImage(cropped)  # z,y,x
        z_mean_top = float(arr[:min(8, arr.shape[0])].mean())
        z_mean_bot = float(arr[-min(8, arr.shape[0]):].mean())
        print(f"[QC] {p.parent.name}: z_top_mean={z_mean_top:.1f}, "
              f"z_bot_mean={z_mean_bot:.1f}, shape={tuple(arr.shape)}")

    print("[DONE] Fixed-center crop complete.")

if __name__ == "__main__":
    main()