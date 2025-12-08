import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from scipy.ndimage import uniform_filter

BASE_DIR = r"E:/LD-CT SR/Data/HCC Abd NC-CT"
OUT_DIR = r"E:/LD-CT SR/Outputs/noise_analysis_corrected"

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


def hanning_2d(shape):
    """2D Hanning window to reduce edge effects"""
    win_y = np.hanning(shape[0])
    win_x = np.hanning(shape[1])
    return np.outer(win_y, win_x)


def extract_roi(hu, mask, roi_size=128):
    """Extract uniform ROI from body region"""
    # Find body region
    coords = np.argwhere(mask > 0.5)
    if len(coords) < roi_size**2:
        return None
    
    # Get center of mass
    cy, cx = coords.mean(axis=0).astype(int)
    
    # Extract ROI
    half = roi_size // 2
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    
    # Bounds check
    if y1 < 0 or x1 < 0 or y2 >= hu.shape[0] or x2 >= hu.shape[1]:
        return None
    
    roi = hu[y1:y2, x1:x2]
    return roi


def compute_nps_single_roi(roi):
    """
    Compute NPS for a single ROI
    Following proper NPS calculation protocol:
    1. Mean subtraction
    2. Window function
    3. FFT
    4. Power spectrum
    5. Normalization
    """
    if roi is None or roi.size == 0:
        return None
    
    # Step 1: Mean subtraction (CRITICAL!)
    roi_centered = roi - np.mean(roi)
    
    # Step 2: Apply Hanning window to reduce edge effects
    window = hanning_2d(roi.shape)
    roi_windowed = roi_centered * window
    
    # Step 3: 2D FFT and shift
    fft_result = fftshift(fft2(roi_windowed))
    
    # Step 4: Power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_result) ** 2
    
    # Step 5: Normalization
    # Normalize by ROI size and window correction factor
    window_correction = np.sum(window**2)
    nps = power_spectrum / window_correction
    
    return nps


def compute_nps_multiple_rois(hu, mask, roi_size=128, n_rois=5):
    """
    Compute NPS by averaging over multiple ROIs
    This reduces statistical variance
    """
    nps_list = []
    
    # Get valid coordinates
    coords = np.argwhere(mask > 0.5)
    if len(coords) < roi_size**2:
        return None
    
    # Randomly sample ROI centers
    np.random.seed(42)
    half = roi_size // 2
    
    attempts = 0
    max_attempts = n_rois * 10
    
    while len(nps_list) < n_rois and attempts < max_attempts:
        attempts += 1
        
        # Random center
        idx = np.random.randint(0, len(coords))
        cy, cx = coords[idx]
        
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        
        # Bounds check
        if y1 < 0 or x1 < 0 or y2 >= hu.shape[0] or x2 >= hu.shape[1]:
            continue
        
        roi = hu[y1:y2, x1:x2]
        roi_mask = mask[y1:y2, x1:x2]
        
        # Check if ROI is mostly within body
        if np.mean(roi_mask) < 0.8:
            continue
        
        # Compute NPS for this ROI
        nps = compute_nps_single_roi(roi)
        if nps is not None:
            nps_list.append(nps)
    
    if len(nps_list) == 0:
        return None
    
    # Average over all ROIs
    nps_avg = np.mean(nps_list, axis=0)
    
    return nps_avg


def compute_nps(hu, mask):
    """
    Main NPS computation function
    Returns properly calculated NPS
    """
    # Use multiple ROI averaging for better statistics
    nps = compute_nps_multiple_rois(hu, mask, roi_size=128, n_rois=5)
    
    if nps is None:
        # Fallback: try single ROI
        roi = extract_roi(hu, mask, roi_size=128)
        nps = compute_nps_single_roi(roi)
    
    return nps


def analyze_directional_nps(nps):
    """방향별 NPS 분석으로 streak 검출"""
    cy, cx = nps.shape[0] // 2, nps.shape[1] // 2
    y, x = np.ogrid[:nps.shape[0], :nps.shape[1]]
    y_centered = y - cy
    x_centered = x - cx
    
    angles = np.arctan2(y_centered, x_centered)
    angles_deg = np.degrees(angles) % 180
    
    # Angular profile
    n_bins = 180
    angular_profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (angles_deg >= i) & (angles_deg < i + 1)
        if mask.sum() > 0:
            angular_profile[i] = nps[mask].mean()
    
    # Directional strengths
    horizontal_mask = ((angles_deg < 10) | (angles_deg > 170))
    horizontal_strength = nps[horizontal_mask].mean() if horizontal_mask.sum() > 0 else 0
    
    vertical_mask = (angles_deg >= 80) & (angles_deg <= 100)
    vertical_strength = nps[vertical_mask].mean() if vertical_mask.sum() > 0 else 0
    
    # Anisotropy
    anisotropy = angular_profile.max() / angular_profile.min() if angular_profile.min() > 0 else np.inf
    
    return angular_profile, horizontal_strength, vertical_strength, anisotropy


def analyze_slice(hu, out_path):
    mask = body_mask(hu)
    noise = compute_noise_std(hu, mask)
    nps = compute_nps(hu, mask)

    # Save visualization
    fig = plt.figure(figsize=(16, 8))
    
    # Layout: 2x3 grid
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(hu, cmap='gray', vmin=-160, vmax=240)
    ax1.set_title(f"HU Image\nNoise: {noise:.2f}")
    ax1.axis('off')

    if nps is not None:
        # NPS 2D
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(np.log1p(nps), cmap='inferno')
        ax2.set_title("NPS (log scale)")
        ax2.axis('off')
        
        # Radial profile
        ax3 = plt.subplot(2, 3, 3)
        cy, cx = nps.shape[0]//2, nps.shape[1]//2
        y, x = np.ogrid[:nps.shape[0], :nps.shape[1]]
        r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
        
        radial_profile = []
        for i in range(int(r.max())):
            mask_ring = (r == i)
            if mask_ring.sum() > 0:
                radial_profile.append(nps[mask_ring].mean())
        
        ax3.plot(radial_profile)
        ax3.set_xlabel('Radial Frequency (pixels)')
        ax3.set_ylabel('NPS Power')
        ax3.set_title('Radial NPS Profile')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Directional analysis
        angular_prof, h_strength, v_strength, anisotropy = analyze_directional_nps(nps)
        
        # Angular profile
        ax4 = plt.subplot(2, 3, 4)
        angles = np.arange(180)
        ax4.plot(angles, angular_prof)
        ax4.axhline(y=h_strength, color='r', linestyle='--', alpha=0.7, label='Horizontal')
        ax4.axhline(y=v_strength, color='b', linestyle='--', alpha=0.7, label='Vertical')
        ax4.set_xlabel('Angle (degrees)')
        ax4.set_ylabel('NPS Power')
        ax4.set_title(f'Angular Profile\nAnisotropy: {anisotropy:.2f}')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Directional strength comparison
        ax5 = plt.subplot(2, 3, 5)
        directions = ['Horizontal', 'Vertical']
        strengths = [h_strength, v_strength]
        bars = ax5.bar(directions, strengths, color=['red', 'blue'], alpha=0.7)
        ax5.set_ylabel('NPS Power')
        ax5.set_title('Directional Strength')
        ax5.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Noise characterization
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Determine noise characteristics
        char_text = "NOISE CHARACTERIZATION:\n\n"
        
        # Low-frequency check
        if len(radial_profile) > 20:
            low_freq_ratio = radial_profile[0] / radial_profile[20]
            if low_freq_ratio > 100:
                char_text += "✓ Strong LOW-FREQ shading\n"
                char_text += f"  Ratio: {low_freq_ratio:.1f}x\n\n"
        
        # Anisotropy check
        if anisotropy > 1.5:
            char_text += "✓ STREAK artifacts detected\n"
            char_text += f"  Anisotropy: {anisotropy:.2f}\n"
            if h_strength > v_strength * 1.2:
                char_text += "  Direction: HORIZONTAL\n\n"
            elif v_strength > h_strength * 1.2:
                char_text += "  Direction: VERTICAL\n\n"
            else:
                char_text += "  Direction: MIXED\n\n"
        elif anisotropy > 1.2:
            char_text += "✓ Mild directional pattern\n"
            char_text += f"  Anisotropy: {anisotropy:.2f}\n\n"
        else:
            char_text += "✓ Mostly ISOTROPIC noise\n\n"
        
        ax6.text(0.05, 0.95, char_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                family='monospace')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return noise


def analyze_patient(patient_dir):
    all_files = [f for f in os.listdir(patient_dir) if f.lower().endswith(".dcm")]

    dcm_infos = []
    for fname in all_files:
        fpath = os.path.join(patient_dir, fname)
        ds = pydicom.dcmread(fpath, stop_before_pixels=True)
        inst = getattr(ds, "InstanceNumber", 0)
        dcm_infos.append((inst, fname))

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
    
    summary_csv = os.path.join(OUT_DIR, "noise_summary_corrected.csv")
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
    print("=== DONE: Corrected NPS analysis completed ===")