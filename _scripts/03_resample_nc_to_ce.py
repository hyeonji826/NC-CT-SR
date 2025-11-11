# -*- coding: utf-8 -*-
"""
Resample NC (NC_crop.nii) -> CE grid (CE_raw.nii.gz)
- 입력 기본: Data/nii_cropped/<id7>/NC_crop.nii
- 타깃:     Data/nii_raw/CE_D/<id7>/CE_raw.nii.gz
- 출력:     Data/nii_preproc/NC/<id7>/NC_resampled.nii.gz
- 로그:     Outputs/reports/resample_log.csv
- pairs:    Data/pairs.csv (있으면 사용, 없으면 트리 스캔)
"""

import csv, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# ---------- defaults ----------
ROOT = Path(r"E:\LD-CT SR")
DIR_NC = "Data/nii_cropped"
DIR_CE = "Data/nii_raw/CE_D"
DIR_OUT = "Data/nii_preproc/NC"
PAIRS_CSV = "Data/pairs.csv"
LOG_CSV = "Outputs/reports/resample_log.csv"

# ---------- utils ----------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def arr(img): return sitk.GetArrayFromImage(img)
def meta(img):
    return {
        "size": "x".join(map(str, img.GetSize())),
        "spacing": "x".join(f"{v:.5f}" for v in img.GetSpacing()),
        "origin": "x".join(f"{v:.3f}" for v in img.GetOrigin()),
        "direction": "x".join(f"{v:.6f}" for v in img.GetDirection()),
    }
def fill_ratio(a: np.ndarray, val: float, atol: float=1e-3)->float:
    return float(np.mean(np.isclose(a, val, atol=atol))) if a.size else 1.0

# === 교체: moments_tx ===
def moments_tx(mov, fix):
    dim_m = mov.GetDimension()
    dim_f = fix.GetDimension()
    if dim_m != dim_f:
        raise RuntimeError(f"dim mismatch: moving={dim_m} vs fixed={dim_f}")
    if dim_m == 2:
        return sitk.CenteredTransformInitializer(
            fix, mov, sitk.Euler2DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
    elif dim_m == 3:
        return sitk.CenteredTransformInitializer(
            fix, mov, sitk.VersorRigid3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
    else:
        raise RuntimeError(f"unsupported dimension: {dim_m}")


# === 교체: refine_translation ===
def refine_translation(mov, fix, init_tx):
    # 정합은 float32로
    movf = sitk.Cast(mov, sitk.sitkFloat32)
    fixf = sitk.Cast(fix, sitk.sitkFloat32)

    dim = mov.GetDimension()
    if dim == 2:
        tx = sitk.Euler2DTransform(init_tx) if isinstance(init_tx, sitk.Euler2DTransform) else sitk.Euler2DTransform()
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 0.1, 40, relaxationFactor=0.5)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetInitialTransform(tx, inPlace=False)
        reg.SetOptimizerWeights([0, 1, 1])  # angle 고정, tx/ty만
        return reg.Execute(fixf, movf)
    elif dim == 3:
        tx = (init_tx if isinstance(init_tx, sitk.VersorRigid3DTransform)
              else sitk.VersorRigid3DTransform())
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(1.0, 0.1, 40, relaxationFactor=0.5)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetInitialTransform(tx, inPlace=False)
        reg.SetOptimizerWeights([0, 0, 0, 1, 1, 1])  # 회전 고정, 이동만
        return reg.Execute(fixf, movf)
    else:
        raise RuntimeError(f"unsupported dimension for refine: {dim}")


def resample_once(mov, fix, tx, outside=-1024.0):
    return sitk.Resample(mov, fix, tx, sitk.sitkLinear, outside, mov.GetPixelID())

# ---------- input discovery ----------
def load_pairs_or_scan(root: Path, pairs_path: Path, nc_root: Path, ce_root: Path):
    rows = []
    if pairs_path.exists():
        df = pd.read_csv(pairs_path)
        key = "id7" if "id7" in df.columns else ("id" if "id" in df.columns else None)
        for _, r in df.iterrows():
            pid = str(r.get(key) or "")
            nc = r.get("input_nc") or r.get("input_nc_crop") or ""
            ce = r.get("target_ce_raw") or r.get("target_ce") or ""
            rows.append((pid, Path(nc), Path(ce)))
    else:
        # scan tree
        for d in sorted([x for x in nc_root.iterdir() if x.is_dir()]):
            pid = d.name
            nc = d / "NC_crop.nii"
            ce = ce_root / pid / "CE_raw.nii.gz"
            rows.append((pid, nc, ce))
    return rows

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="NC -> CE resample (robust)")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--nc_dir", default=DIR_NC)
    ap.add_argument("--ce_dir", default=DIR_CE)
    ap.add_argument("--out_dir", default=DIR_OUT)
    ap.add_argument("--pairs", default=PAIRS_CSV)
    ap.add_argument("--log", default=LOG_CSV)
    ap.add_argument("--outside", type=float, default=-1024.0)
    ap.add_argument("--reorient-lps", action="store_true")
    ap.add_argument("--no-rigid", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    nc_root = root / args.nc_dir
    ce_root = root / args.ce_dir
    out_root = root / args.out_dir
    log_csv = root / args.log
    ensure_dir(log_csv.parent)

    pairs_path = root / args.pairs
    cases = load_pairs_or_scan(root, pairs_path, nc_root, ce_root)
    if not cases:
        print("[ERR] no cases found")
        return

    logs = []
    for pid_raw, nc_path, ce_path in tqdm(cases, desc="Resample NC→CE", ncols=100):
        try:
            # id 7자리 정규화
            pid7 = str(pid_raw).strip()
            if pid7.isdigit():
                pid7 = pid7.zfill(7)

            # 입력 존재
            if not nc_path.exists() or not ce_path.exists():
                logs.append({"id7": pid7, "status": "missing_input",
                             "nc_path": str(nc_path), "ce_path": str(ce_path)})
                continue

            # 로드
            mov = sitk.ReadImage(str(nc_path))  # NC
            fix = sitk.ReadImage(str(ce_path))  # CE

            # 차원 불일치 즉시 스킵 (여기서 걸러야 moments 예외 안 남)
            if mov.GetDimension() != fix.GetDimension():
                logs.append({"id7": pid7,
                             "status": f"dim_mismatch:{mov.GetDimension()}vs{fix.GetDimension()}",
                             "nc_path": str(nc_path), "ce_path": str(ce_path)})
                continue

            # 필요 시 LPS 재배향
            if args.reorient_lps:
                try:
                    ori = sitk.DICOMOrientImageFilter(); ori.SetDesiredCoordinateOrientation("LPS")
                    mov = ori.Execute(mov); fix = ori.Execute(fix)
                except Exception:
                    pass

            # 정합/품질 판단은 float32로 계산
            movf = sitk.Cast(mov, sitk.sitkFloat32)
            fixf = sitk.Cast(fix, sitk.sitkFloat32)

            # 1) identity
            id_tx = sitk.Transform(mov.GetDimension(), sitk.sitkIdentity)
            resf = resample_once(movf, fixf, id_tx, outside=args.outside)
            r_np = arr(resf); fr = fill_ratio(r_np, args.outside)
            used_rigid = 0; used_refine = 0

            # 2) moments rigid (거의 바깥값이면)
            if (not args.no_rigid) and fr > 0.95:
                init_tx = moments_tx(movf, fixf)  # ✅ float32 기반, 2D/3D 분기
                resf2 = resample_once(movf, fixf, init_tx, outside=args.outside)
                r2_np = arr(resf2); fr2 = fill_ratio(r2_np, args.outside)
                resf, r_np, fr = resf2, r2_np, fr2
                used_rigid = 1

                # 3) translation refine
                if (not args.no_refine) and fr2 > 0.50:
                    try:
                        tx = refine_translation(mov, fix, init_tx)  # 내부에서 float32 정합
                        resf3 = resample_once(movf, fixf, tx, outside=args.outside)
                        r3_np = arr(resf3); fr3 = fill_ratio(r3_np, args.outside)
                        if fr3 < fr2:
                            resf, r_np, fr = resf3, r3_np, fr3
                            used_refine = 1
                    except Exception as e:
                        logs.append({"id7": pid7, "status": f"refine_error:{e}"})

            # 최종 저장은 원본 픽셀타입으로 리샘플해서 기록 (헤더/그리드는 CE 기준)
            final_tx = sitk.Transform(mov.GetDimension(), sitk.sitkIdentity)
            # moments/refine를 썼다면 그 Transform을 사용
            if used_rigid:
                final_tx = moments_tx(movf, fixf)
                if used_refine:
                    try:
                        final_tx = refine_translation(mov, fix, final_tx)
                    except Exception:
                        pass
            res_save = resample_once(mov, fix, final_tx, outside=args.outside)

            out_path = out_root / pid7 / "NC_resampled.nii.gz"
            ensure_dir(out_path.parent)
            sitk.WriteImage(res_save, str(out_path), useCompression=True)

            # QC
            p1, p99 = (np.percentile(r_np, [1, 99]) if r_np.size else (-1e9, -1e9))
            status = "success"
            if fr > 0.95 or (abs(p1 - args.outside) < 5 and abs(p99 - args.outside) < 5):
                status = "empty_after_resample"

            logs.append({
                "id7": pid7,
                "nc_path": str(nc_path), "ce_path": str(ce_path),
                "out_path": str(out_path),
                "ce_meta": meta(fix),
                "nc_resampled_meta": meta(res_save),
                "fill_ratio": f"{fr:.6f}",
                "p1": f"{float(p1):.3f}", "p99": f"{float(p99):.3f}",
                "used_rigid": used_rigid, "used_refine": used_refine,
                "status": status
            })

        except Exception as e:
            logs.append({"id7": str(pid_raw), "status": f"case_error:{e}",
                         "nc_path": str(nc_path), "ce_path": str(ce_path)})
            continue

    with log_csv.open("w", newline="", encoding="utf-8") as f:
        fields = sorted({k for d in logs for k in d.keys()})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(logs)
    print(f"[OK] log -> {log_csv}")

if __name__ == "__main__":
    main()
