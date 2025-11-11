"""
pairs.csv QC:
- 존재 여부, 로드 가능 여부
- size/spacing/direction 동일성(허용오차)
- origin 차이(mm) 요약
결과: Outputs/reports/pairs_qc.csv
"""

import csv
import math
from pathlib import Path
import argparse

import SimpleITK as sitk
from tqdm import tqdm

DEF_PAIRS = r"E:\LD-CT SR\Data\pairs.csv"
DEF_OUT   = r"E:\LD-CT SR\Outputs\reports\pairs_qc.csv"

def read_pairs(pairs_csv: Path):
    rows = []
    with pairs_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def load_meta(p: Path):
    meta = {
        "ok": False, "size": "", "spacing": "", "direction": "", "origin": "",
        "size_xyz": None, "spacing_xyz": None, "direction_vec": None, "origin_xyz": None
    }
    try:
        img = sitk.ReadImage(str(p))
        size = img.GetSize()             # x, y, z
        spacing = img.GetSpacing()       # x, y, z
        direction = img.GetDirection()   # 9 values (3x3)
        origin = img.GetOrigin()         # x, y, z

        meta["ok"] = True
        meta["size_xyz"] = tuple(map(int, size))
        meta["spacing_xyz"] = tuple(map(float, spacing))
        meta["direction_vec"] = tuple(map(float, direction))
        meta["origin_xyz"] = tuple(map(float, origin))

        meta["size"] = "x".join(map(str, size))
        meta["spacing"] = "x".join(f"{s:.3f}" for s in spacing)
        meta["direction"] = "x".join(f"{d:.6f}" for d in direction)
        meta["origin"] = "x".join(f"{o:.3f}" for o in origin)
    except Exception:
        pass
    return meta

def vec_close(a, b, eps):
    if a is None or b is None: return False
    if len(a) != len(b): return False
    return all(abs(a[i]-b[i]) <= eps for i in range(len(a)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=DEF_PAIRS, type=str, help="Path to Data/pairs.csv")
    ap.add_argument("--out", default=DEF_OUT, type=str, help="Output CSV path")
    ap.add_argument("--eps-spacing", default=0.01, type=float, help="mm tolerance for spacing equality")
    ap.add_argument("--eps-dir", default=1e-4, type=float, help="tolerance for direction cosines")
    ap.add_argument("--report-missing-only", action="store_true", help="Keep only rows with any issues")
    args = ap.parse_args()

    pairs_csv = Path(args.pairs)
    out_csv   = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pairs = read_pairs(pairs_csv)
    if not pairs:
        print(f"[ERR] Empty or invalid pairs: {pairs_csv}")
        return

    results = []
    issues = 0

    for row in tqdm(pairs, desc="QC pairs", ncols=100):
        pid = row.get("id", "")
        nc_p = Path(row.get("input_nc", ""))
        ce_p = Path(row.get("target_ce", row.get("target_ce_gridmatch", "")))

        exists_nc = nc_p.exists()
        exists_ce = ce_p.exists()

        meta_nc = load_meta(nc_p) if exists_nc else {}
        meta_ce = load_meta(ce_p) if exists_ce else {}

        load_nc = bool(meta_nc.get("ok", False))
        load_ce = bool(meta_ce.get("ok", False))

        # comparisons
        size_equal = (meta_nc.get("size_xyz") == meta_ce.get("size_xyz")) if (load_nc and load_ce) else False
        spacing_equal = vec_close(meta_nc.get("spacing_xyz"), meta_ce.get("spacing_xyz"), args.eps_spacing) if (load_nc and load_ce) else False
        direction_equal = vec_close(meta_nc.get("direction_vec"), meta_ce.get("direction_vec"), args.eps_dir) if (load_nc and load_ce) else False

        origin_delta = ""
        if load_nc and load_ce:
            on = meta_nc["origin_xyz"]; oc = meta_ce["origin_xyz"]
            d = math.sqrt((on[0]-oc[0])**2 + (on[1]-oc[1])**2 + (on[2]-oc[2])**2)
            origin_delta = f"{d:.2f}mm"

        has_issue = (not exists_nc) or (not exists_ce) or (not load_nc) or (not load_ce) \
                    or (not spacing_equal) or (not direction_equal)

        if has_issue: issues += 1

        results.append({
            "id": pid,
            "input_nc": str(nc_p),
            "target_ce": str(ce_p),
            "exists_nc": int(exists_nc),
            "exists_ce": int(exists_ce),
            "load_nc": int(load_nc),
            "load_ce": int(load_ce),
            "nc_size": meta_nc.get("size",""),
            "ce_size": meta_ce.get("size",""),
            "size_equal": int(size_equal),
            "nc_spacing": meta_nc.get("spacing",""),
            "ce_spacing": meta_ce.get("spacing",""),
            "spacing_equal(<=eps)": int(spacing_equal),
            "nc_direction": meta_nc.get("direction",""),
            "ce_direction": meta_ce.get("direction",""),
            "direction_equal(<=eps)": int(direction_equal),
            "nc_origin": meta_nc.get("origin",""),
            "ce_origin": meta_ce.get("origin",""),
            "origin_delta_mm(L2)": origin_delta,
            "issue": int(has_issue)
        })

    # 선택: 이슈만 남기기
    if args.report_missing_only:
        results = [r for r in results if r["issue"] == 1]

    # write
    if results:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    total = len(pairs)
    print(f"[SUMMARY] total={total}, issues={issues}, clean={total-issues}")
    print(f"[OK] report -> {out_csv}")

if __name__ == "__main__":
    main()
