"""
SimpleITK 기반 '물리좌표 크롭' 최종본
- DICOM -> NIfTI 변환(짧은 파일명)
- CE 바디마스크로 bbox 산출 -> crop_info.json 저장 (없으면 생성)
- crop_info.json의 bbox(CE index, ZYX 순서)를 '물리좌표'로 바꿔 NC에 투영하여 크롭
- tqdm 진행률 표시

필요 패키지: SimpleITK, numpy, tqdm
    pip install SimpleITK numpy tqdm
"""

import os, re, json, csv
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ===== 경로 설정 =====
ROOT = Path(r"E:\LD-CT SR")
DATA_DIR = ROOT / "Data"

DCM_NC_ROOT = DATA_DIR / "HCC Abd NC-CT"
DCM_CE_ROOT = DATA_DIR / "HCC CE-CT (D)"

OUT_NII_RAW = DATA_DIR / "nii_raw"
OUT_NII_RAW_NC = OUT_NII_RAW / "NC"
OUT_NII_RAW_CE = OUT_NII_RAW / "CE_D"
OUT_NII_CROP = DATA_DIR / "nii_cropped" 
OUT_META = DATA_DIR / "crop_info"
PAIRS_CSV = DATA_DIR / "pairs.csv"

for d in [OUT_NII_RAW_NC, OUT_NII_RAW_CE, OUT_NII_CROP, OUT_META]:
    d.mkdir(parents=True, exist_ok=True)

# ===== 유틸 =====
def extract_id(name: str):
    m = re.search(r"(\d{5,8})", name)
    return m.group(1) if m else re.sub(r"[^A-Za-z0-9]", "_", name)

def find_dcm_leaf_dirs(root: Path, max_depth=2):
    """루트 하위 depth<=2 폴더 중 .dcm이 있는 폴더 수집"""
    if not root.exists():
        return []
    leafs = []
    for p in root.rglob("*"):
        try:
            if p.is_dir() and p.relative_to(root).parts and len(p.relative_to(root).parts) <= max_depth:
                if next(p.glob("*.dcm"), None) is not None:
                    leafs.append(p)
        except Exception:
            pass
    return sorted(set(leafs))

def read_one_series(folder: Path) -> sitk.Image:
    """가장 슬라이스 많은 시리즈 하나 읽기"""
    reader = sitk.ImageSeriesReader()
    sids = reader.GetGDCMSeriesIDs(str(folder))
    if not sids:
        return None
    def num_files(sid):
        return len(reader.GetGDCMSeriesFileNames(str(folder), sid))
    sid = sorted(sids, key=num_files, reverse=True)[0]
    files = reader.GetGDCMSeriesFileNames(str(folder), sid)
    reader.SetFileNames(files)
    return reader.Execute()

def save_nifti(img: sitk.Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)

def largest_component(mask: sitk.Image) -> sitk.Image:
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    if stats.GetNumberOfLabels() == 0:
        return mask
    largest = max(stats.GetLabels(), key=lambda l: stats.GetPhysicalSize(l))
    return sitk.Equal(cc, largest)

def body_mask_from_hu(ce_img: sitk.Image, hu_min=-200) -> sitk.Image:
    mask = sitk.Greater(ce_img, hu_min)
    mask = sitk.BinaryMorphologicalClosing(mask, [3,3,3])
    mask = sitk.BinaryFillhole(mask)
    mask = largest_component(mask)
    return sitk.Cast(mask, sitk.sitkUInt8)

def bbox_index_zyx(mask: sitk.Image) -> Tuple[int,int,int,int,int,int]:
    arr = sitk.GetArrayFromImage(mask)  # z,y,x
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        Z,Y,X = arr.shape
        return (0,Z, 0,Y, 0,X)
    z0,y0,x0 = coords.min(axis=0)
    z1,y1,x1 = coords.max(axis=0) + 1
    return int(z0),int(z1),int(y0),int(y1),int(x0),int(x1)

def write_crop_info_json(pid: str, ce_img: sitk.Image, bbox_zyx, out_dir=OUT_META) -> Path:
    info = {
        "id": pid,
        "bbox_index_zyx": {"z0":bbox_zyx[0],"z1":bbox_zyx[1],
                           "y0":bbox_zyx[2],"y1":bbox_zyx[3],
                           "x0":bbox_zyx[4],"x1":bbox_zyx[5]},
        "reference": {
            "size_xyz": list(map(int, ce_img.GetSize())),
            "spacing_xyz": list(map(float, ce_img.GetSpacing())),
            "origin_xyz": list(map(float, ce_img.GetOrigin())),
            "direction": list(map(float, ce_img.GetDirection()))
        }
    }
    out_path = out_dir / f"{pid}_crop_info.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return out_path

def load_crop_info_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def bbox_physical_from_ceinfo(ce_ref: sitk.Image, bbox_zyx) -> Tuple[Tuple[float,float,float], Tuple[float,float,float]]:
    """CE bbox(zyx) -> 물리좌표(min_phys, max_phys) 구하기"""
    z0,z1,y0,y1,x0,x1 = bbox_zyx
    # CE index는 x,y,z 순서로 넣어야 함
    p0 = ce_ref.TransformIndexToPhysicalPoint((int(x0), int(y0), int(z0)))
    p1 = ce_ref.TransformIndexToPhysicalPoint((int(x1)-1, int(y1)-1, int(z1)-1))
    # 각 축 min/max 정렬
    min_phys = tuple(min(a,b) for a,b in zip(p0,p1))
    max_phys = tuple(max(a,b) for a,b in zip(p0,p1))
    return min_phys, max_phys

def crop_nc_by_physical_box(nc_img: sitk.Image, min_phys, max_phys) -> sitk.Image:
    """물리좌표 박스를 NC 인덱스 범위로 변환하여 ROI"""
    # 물리좌표 경계 → index 경계 (x,y,z)
    i0 = list(nc_img.TransformPhysicalPointToIndex(min_phys))
    i1 = list(nc_img.TransformPhysicalPointToIndex(max_phys))
    # 보정(작은게 start)
    start = [min(i0[0], i1[0]), min(i0[1], i1[1]), min(i0[2], i1[2])]
    stop  = [max(i0[0], i1[0])+1, max(i0[1], i1[1])+1, max(i0[2], i1[2])+1]
    # 이미지 범위로 클램프
    size_xyz = list(nc_img.GetSize())
    start = [max(0, s) for s in start]
    stop  = [min(size_xyz[d], stop[d]) for d in range(3)]
    out_size = [max(0, stop[d] - start[d]) for d in range(3)]
    if 0 in out_size:
        raise RuntimeError("Cropping box outside NC image (no overlap).")
    roi = sitk.RegionOfInterest(nc_img, out_size, start)
    return roi

# ===== 메인 파이프라인 =====
def main():
    # 1) DICOM 폴더 수집 & ID 매칭
    nc_dirs = find_dcm_leaf_dirs(DCM_NC_ROOT)
    ce_dirs = find_dcm_leaf_dirs(DCM_CE_ROOT)
    id_nc = {extract_id(p.name): p for p in nc_dirs}
    id_ce = {extract_id(p.name): p for p in ce_dirs}
    ids = sorted(set(id_nc) & set(id_ce))

    print(f"[INFO] matched IDs: {len(ids)}")

    rows = []
    for pid in tqdm(ids, desc="Convert & Crop", ncols=100):
        ce_dir = id_ce[pid]
        nc_dir = id_nc[pid]

        # ---- CE 읽기 → 저장 (짧은 파일명)
        ce_img = read_one_series(ce_dir)
        if ce_img is None:
            print(f"[WARN] CE read fail: {pid}"); continue
        ce_out = OUT_NII_RAW_CE / pid / "CE_raw.nii.gz"
        save_nifti(ce_img, ce_out)

        # ---- NC 읽기 → 저장
        nc_img = read_one_series(nc_dir)
        if nc_img is None:
            print(f"[WARN] NC read fail: {pid}"); continue
        nc_out = OUT_NII_RAW_NC / pid / "NC_raw.nii.gz"
        save_nifti(nc_img, nc_out)

        # ---- crop_info 확보: 있으면 재사용, 없으면 생성
        info_path = OUT_META / f"{pid}_crop_info.json"
        if info_path.exists():
            info = load_crop_info_json(info_path)
            z0 = info["bbox_index_zyx"]["z0"]; z1 = info["bbox_index_zyx"]["z1"]
            y0 = info["bbox_index_zyx"]["y0"]; y1 = info["bbox_index_zyx"]["y1"]
            x0 = info["bbox_index_zyx"]["x0"]; x1 = info["bbox_index_zyx"]["x1"]
            bbox_zyx = (z0,z1,y0,y1,x0,x1)
            # 참고: 방향/원점이 다른 CE일 수도 있으니, 안전하게 현재 ce_img로 물리 변환 수행
        else:
            # CE 바디마스크로 bbox 생성
            ce_mask = body_mask_from_hu(ce_img, hu_min=-200)
            bbox_zyx = bbox_index_zyx(ce_mask)
            info_path = write_crop_info_json(pid, ce_img, bbox_zyx)

        # ---- CE bbox(인덱스, zyx) → 물리좌표 박스
        min_phys, max_phys = bbox_physical_from_ceinfo(ce_img, bbox_zyx)

        # ---- 물리좌표 박스로 NC 크롭 (리샘플 X, 회색화면 문제 근본 해결)
        try:
            nc_crop = crop_nc_by_physical_box(nc_img, min_phys, max_phys)
        except RuntimeError as e:
            print(f"[WARN] {pid}: {e}")
            continue

        crop_out = OUT_NII_CROP / f"{pid}_NC_crop.nii.gz"
        save_nifti(nc_crop, crop_out)

        rows.append({
            "id": pid,
            "ce_path": str(ce_out),
            "nc_path": str(nc_out),
            "nc_cropped_path": str(crop_out),
            "crop_info_json": str(info_path)
        })

    if rows:
        with open(PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("[DONE] Physical-space crop pipeline complete.")

if __name__ == "__main__":
    main()
