# _scripts_4_wavelet/analyze_noise.py

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pydicom
import nibabel as nib
from scipy.ndimage import gaussian_filter
import pandas as pd


# ========= 설정 =========
# 이 스크립트 파일 위치 기준으로 path 잡기
SCRIPT_DIR = Path(__file__).parent
ROOT = SCRIPT_DIR.parent  # LD-CT SR/
DATA_DIR = ROOT / "Data"

DCM_ROOT = DATA_DIR / "HCC Abd NC-CT"
NII_ROOT = DATA_DIR / "Image_NC-CT"

# 전처리에서 쓴 HU window (config_n2n.yaml 과 동일해야 함)
HU_MIN, HU_MAX = -160.0, 240.0


# ========= 유틸 함수들 =========

def load_dcm_series(case_dir: Path) -> List[np.ndarray]:
    """해당 케이스 DICOM 시리즈 전체를 HU로 로드해서 리스트로 반환."""
    if not case_dir.exists():
        return []

    dcm_files = sorted(
        [p for p in case_dir.glob("*.dcm")],
        key=lambda p: p.name
    )
    slices = []
    for fp in dcm_files:
        ds = pydicom.dcmread(str(fp))
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        slices.append(hu)

    return slices


def load_nii_volume(nii_path: Path) -> np.ndarray:
    """NIfTI 볼륨을 로드해서 (H, W, D) 형태의 HU 이미지로 반환."""
    nii = nib.load(str(nii_path))
    arr = nii.get_fdata().astype(np.float32)

    # (H, W, D[, C]) 형태 가정
    if arr.ndim == 4:
        arr = arr[..., 0]

    if arr.ndim != 3:
        raise ValueError(f"Unexpected NIfTI shape: {arr.shape}")

    # ✅ 이미 HU 로 저장되어 있으므로 그대로 사용
    #  (추가 스케일링 금지)
    return arr  # (H, W, D)


def compute_noise_stats(img_hu: np.ndarray) -> Dict:
    """
    2D HU 이미지에서 영역별 노이즈 통계 계산.
    air / lung / soft / bone + 자기상관
    """
    stats: Dict = {}

    # 마스크 정의
    air_mask = img_hu < -900
    lung_mask = (img_hu > -900) & (img_hu < -400)
    soft_mask = (img_hu > -100) & (img_hu < 100)
    bone_mask = img_hu > 300

    masks = {
        "air": air_mask,
        "lung": lung_mask,
        "soft": soft_mask,
        "bone": bone_mask,
    }

    # 하이패스 (구조 제거 후 high-frequency component)
    lp = gaussian_filter(img_hu, sigma=1.0)
    hp = img_hu - lp

    for key, m in masks.items():
        if m.sum() < 200:
            continue

        vals_raw = img_hu[m]
        vals_hp = hp[m]

        mean_hu = float(vals_raw.mean())
        std_raw = float(vals_raw.std())
        std_hp = float(vals_hp.std())
        ratio = std_hp / std_raw if std_raw > 0 else 0.0

        stats[key] = {
            "pixels": int(m.sum()),
            "mean_hu": mean_hu,
            "std_raw": std_raw,
            "std_hp": std_hp,
            "hp_over_raw": ratio,
        }

    # 중심 ROI에서 row/col 자기상관 (streak 여부)
    def autocorr_1d(arr):
        arr = arr.astype(np.float32)
        arr = arr - arr.mean()
        num = float(np.sum(arr[:-1] * arr[1:]))
        den = float(np.sum(arr ** 2) + 1e-8)
        return num / den

    H, W = img_hu.shape
    h0, h1 = H // 4, 3 * H // 4
    w0, w1 = W // 4, 3 * W // 4
    center_roi = img_hu[h0:h1, w0:w1]

    row_corr = autocorr_1d(center_roi.mean(axis=0))
    col_corr = autocorr_1d(center_roi.mean(axis=1))

    stats["autocorr"] = {
        "row_mean_corr": row_corr,
        "col_mean_corr": col_corr,
    }

    return stats


def sample_slice_indices(n_slices: int, n_samples: int = 3) -> List[int]:
    """볼륨에서 n_samples 개의 슬라이스 index를 고르게 샘플링."""
    if n_slices <= n_samples:
        return list(range(n_slices))
    idxs = np.linspace(0, n_slices - 1, n_samples)
    return [int(round(i)) for i in idxs]


# ========= 메인 분석 루프 =========

def analyze_dataset():
    records = []

    nii_files = sorted(NII_ROOT.glob("*.nii*"))
    print(f"Found {len(nii_files)} NIfTI volumes in {NII_ROOT}")

    for nii_fp in nii_files:
        # 파일명: 0025980_0000.nii.gz → case_id = 0025980
        stem = nii_fp.name.split("_")[0]
        case_id = stem

        dcm_case_dir = DCM_ROOT / case_id
        if not dcm_case_dir.exists():
            print(f"[WARN] DICOM folder not found for case {case_id}")
            continue

        try:
            vol_nii_hu = load_nii_volume(nii_fp)            # (H, W, D)
            dcm_slices = load_dcm_series(dcm_case_dir)      # list of (H, W)
        except Exception as e:
            print(f"[ERROR] Failed to load {case_id}: {e}")
            continue

        if len(dcm_slices) == 0:
            print(f"[WARN] No DICOM slices for {case_id}")
            continue

        Hn, Wn, Dn = vol_nii_hu.shape
        Dd = len(dcm_slices)
        D = min(Dn, Dd)

        if D == 0:
            print(f"[WARN] Empty volume for {case_id}")
            continue

        slice_indices = sample_slice_indices(D, n_samples=3)

        for idx in slice_indices:
            nii_slice = vol_nii_hu[:, :, idx]
            dcm_slice = dcm_slices[idx]

            # 크기가 다르면 중앙 crop/resize 같은 걸 해야 하지만
            # 보통은 동일하게 나올 거라 가정
            if nii_slice.shape != dcm_slice.shape:
                # 안전하게 중앙부만 맞춰서 비교
                H = min(nii_slice.shape[0], dcm_slice.shape[0])
                W = min(nii_slice.shape[1], dcm_slice.shape[1])
                nii_sub = nii_slice[:H, :W]
                dcm_sub = dcm_slice[:H, :W]
            else:
                nii_sub = nii_slice
                dcm_sub = dcm_slice

            stats_dcm = compute_noise_stats(dcm_sub)
            stats_nii = compute_noise_stats(nii_sub)

            # 영역별로 record 추가
            for src, stats in [("dcm", stats_dcm), ("nii", stats_nii)]:
                for region in ["air", "lung", "soft", "bone"]:
                    if region not in stats:
                        continue
                    r = stats[region]
                    rec = {
                        "case_id": case_id,
                        "slice_idx": int(idx),
                        "source": src,          # dcm or nii
                        "region": region,
                        "pixels": r["pixels"],
                        "mean_hu": r["mean_hu"],
                        "std_raw": r["std_raw"],
                        "std_hp": r["std_hp"],
                        "hp_over_raw": r["hp_over_raw"],
                    }
                    records.append(rec)

                # autocorr 한 번만
                ac = stats.get("autocorr", None)
                if ac is not None:
                    rec_ac = {
                        "case_id": case_id,
                        "slice_idx": int(idx),
                        "source": src,
                        "region": "autocorr",
                        "pixels": 0,
                        "mean_hu": 0.0,
                        "std_raw": 0.0,
                        "std_hp": 0.0,
                        "hp_over_raw": 0.0,
                        "row_mean_corr": ac["row_mean_corr"],
                        "col_mean_corr": ac["col_mean_corr"],
                    }
                    records.append(rec_ac)

        print(f"[OK] analyzed case {case_id} (slices: {D})")

    # DataFrame으로 저장
    if not records:
        print("No records collected.")
        return

    df = pd.DataFrame(records)

    out_dir = SCRIPT_DIR / "noise_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "noise_stats.csv"
    json_path = out_dir / "noise_stats_summary.json"

    df.to_csv(csv_path, index=False)
    print(f"Saved detailed stats to: {csv_path}")

    # 간단 요약도 같이 저장 (region, source 별 mean/std)
    summary = {}
    for (src, region), sub in df.groupby(["source", "region"]):
        summary.setdefault(src, {})[region] = {
            "mean_std_raw": float(sub["std_raw"].mean()),
            "mean_std_hp": float(sub["std_hp"].mean()),
            "mean_hp_over_raw": float(sub["hp_over_raw"].mean()),
            "mean_mean_hu": float(sub["mean_hu"].mean()),
        }

    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to: {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    analyze_dataset()
