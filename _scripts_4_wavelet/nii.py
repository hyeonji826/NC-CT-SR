import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.ndimage import gaussian_filter

BASE_DIR = r"E:/LD-CT SR/Data/HCC Abd NC-CT"
OUT_DIR = r"E:/LD-CT SR/Outputs/noise_analysis"

os.makedirs(OUT_DIR, exist_ok=True)


def load_dcm(path):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    hu = img * ds.RescaleSlope + ds.RescaleIntercept
    return hu


def body_mask(hu):
    mask = (hu > -300) & (hu < 300)
    return mask.astype(np.float32)


def compute_noise_std(hu, mask):
    vals = hu[mask > 0.5]
    if vals.size < 100:
        return np.nan
    return np.std(vals)


def compute_nps(hu, mask):
    roi = hu * mask
    roi = gaussian_filter(roi, 1)
    f = fftshift(np.abs(fft2(roi)))
    return f


def analyze_slice(hu, out_path):
    mask = body_mask(hu)
    noise = compute_noise_std(hu, mask)
    nps = compute_nps(hu, mask)

    # Save visualization
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.imshow(hu, cmap='gray')
    plt.title(f"HU Image\nNoise: {noise:.2f}")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(np.log1p(nps), cmap='inferno')
    plt.title("NPS (log scale)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return noise


def analyze_patient(patient_dir):
    # 1) 폴더 안의 모든 dcm 파일 찾기
    all_files = [f for f in os.listdir(patient_dir) if f.lower().endswith(".dcm")]

    # 2) InstanceNumber 읽어서 정렬 기준 만들기
    dcm_infos = []
    for fname in all_files:
        fpath = os.path.join(patient_dir, fname)
        ds = pydicom.dcmread(fpath, stop_before_pixels=True)  # 속도 위해 픽셀은 안 읽음
        inst = getattr(ds, "InstanceNumber", 0)
        dcm_infos.append((inst, fname))

    # 3) InstanceNumber 기준으로 정렬
    dcm_infos.sort(key=lambda x: x[0])

    results = []
    for inst, fname in dcm_infos:
        fpath = os.path.join(patient_dir, fname)
        hu = load_dcm(fpath)

        out_img = os.path.join(
            OUT_DIR,
            f"{os.path.basename(patient_dir)}_{fname.replace('.dcm','.png')}"
        )

        noise = analyze_slice(hu, out_img)
        results.append([fname, noise])

    return results


def run_all():
    patients = sorted(os.listdir(BASE_DIR))
    
    summary_csv = os.path.join(OUT_DIR, "noise_summary.csv")
    with open(summary_csv, "w") as f:
        f.write("patient,slice,noise_std\n")

        for pid in patients:
            patient_dir = os.path.join(BASE_DIR, pid)
            if not os.path.isdir(patient_dir):
                continue

            print(f"[INFO] Processing patient: {pid}")
            res = analyze_patient(patient_dir)

            for fname, noise in res:
                f.write(f"{pid},{fname},{noise}\n")


if __name__ == "__main__":
    run_all()
    print("=== DONE: All DICOM noise analysis completed ===")
