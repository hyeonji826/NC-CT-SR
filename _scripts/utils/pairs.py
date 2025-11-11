import csv, re
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm

ROOT = Path(r"E:\LD-CT SR")
DATA = ROOT / "Data"
NC_DIR = DATA / "nii_cropped"
CE_DIR = DATA / "nii_raw" / "CE_D"
OUT_CSV = DATA / "pairs.csv"

def id7(name: str):
    m = re.search(r"(\d{5,8})", name)
    return m.group(1) if m else None

def meta(path: Path):
    try:
        img = sitk.ReadImage(str(path))
        return {
            "size_xyz": "x".join(map(str, img.GetSize())),
            "spacing_xyz": "x".join(f"{s:.3f}" for s in img.GetSpacing()),
            "direction": "x".join(f"{d:.3f}" for d in img.GetDirection()),
            "origin_xyz": "x".join(f"{o:.1f}" for o in img.GetOrigin()),
        }
    except Exception:
        return {"size_xyz":"","spacing_xyz":"","direction":"","origin_xyz":""}

def main():
    rows = []
    nc_patients = sorted([p for p in NC_DIR.glob("*") if p.is_dir()])
    print(f"[INFO] candidates: {len(nc_patients)}")

    for d in tqdm(nc_patients, desc="Scan", ncols=100):
        pid = id7(d.name)
        if not pid: 
            print(f"[WARN] skip (no id7): {d.name}")
            continue
        nc_path = d / "NC_crop.nii"
        ce_path = CE_DIR / pid / "CE_raw.nii.gz"

        if not nc_path.exists():
            print(f"[WARN] NC missing: {nc_path}")
            continue
        if not ce_path.exists():
            print(f"[WARN] CE missing: {ce_path}")
            continue

        m_nc = meta(nc_path)
        m_ce = meta(ce_path)

        rows.append({
            "id": pid,
            "input_nc": str(nc_path),
            "target_ce": str(ce_path),
            "nc_size": m_nc["size_xyz"],
            "ce_size": m_ce["size_xyz"],
            "nc_spacing": m_nc["spacing_xyz"],
            "ce_spacing": m_ce["spacing_xyz"],
            "nc_direction": m_nc["direction"],
            "ce_direction": m_ce["direction"],
            "nc_origin": m_nc["origin_xyz"],
            "ce_origin": m_ce["origin_xyz"],
        })

    if rows:
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] pairs.csv -> {OUT_CSV}")
    else:
        print("[ERR] no valid pairs found.")

if __name__ == "__main__":
    main()