# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r"E:/LD-CT SR")
    ap.add_argument("--nc_dir", default="Data/nii_cropped")
    ap.add_argument("--ce_dir", default="Data/nii_raw/CE_D")
    ap.add_argument("--out_csv", default="Data/pairs.csv")
    args = ap.parse_args()

    root = Path(args.root)
    nc_root = root / args.nc_dir
    ce_root = root / args.ce_dir
    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for id_dir in sorted([d for d in nc_root.iterdir() if d.is_dir()]):
        id7 = id_dir.name
        nc = id_dir / "NC_crop.nii"
        ce = ce_root / id7 / "CE_raw.nii.gz"
        if nc.exists() and ce.exists():
            rows.append({"id7": id7, "input_nc": str(nc), "target_ce_raw": str(ce)})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id7","input_nc","target_ce_raw"])
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] pairs -> {out_csv} (rows={len(rows)})")

if __name__ == "__main__":
    main()
