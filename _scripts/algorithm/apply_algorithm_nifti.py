"""
CT Denoising Pipeline - Action-Guided Mask 기반
================================================
NIfTI 직접 로드 → Action mask 생성 → Mask-guided denoising

핵심 원칙:
- 블러링/스무딩 절대 금지 (Edge-preserving만)
- Action mask가 각 영역의 처리 방법을 가이드
  - structure mask → 보존 강도
  - artifact mask → artifact 제거 강도
  - fluid mask → 유체 보존 (denoise 하지 않음)
  - noise_level → denoise 강도

의존: action_guided_masks.py (같은 디렉토리)
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.signal import medfilt2d
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# action_guided_masks 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from action_guided_masks import generate_action_masks, TISSUE_NAMES, TISSUE_COLORS


# ============================================================
# Wavelet Denoising (Edge-preserving, NO blur)
# ============================================================

def wavelet_denoise(image, threshold_scale=3.0, level=2):
    """
    Haar Wavelet Thresholding - Edge-preserving
    블러링 없이 고주파 노이즈만 제거
    """
    import pywt

    coeffs = pywt.wavedec2(image, 'haar', level=level)

    # Estimate noise from finest level
    detail = coeffs[-1]
    sigma = np.median(np.abs(detail[0])) / 0.6745
    threshold = threshold_scale * sigma

    # Soft threshold on detail coefficients only
    new_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        new_detail = tuple(pywt.threshold(d, threshold, mode='soft') for d in detail)
        new_coeffs.append(new_detail)

    denoised = pywt.waverec2(new_coeffs, 'haar')
    if denoised.shape != image.shape:
        denoised = denoised[:image.shape[0], :image.shape[1]]

    return denoised


# ============================================================
# Median Filter (Edge-preserving, NO blur)
# ============================================================

def median_denoise(image, kernel_size=3):
    """Median filter - 엣지 보존, 블러 아님"""
    return medfilt2d(image.astype(np.float32), kernel_size=kernel_size)


# ============================================================
# Action-Guided Adaptive Denoising
# ============================================================

def inpaint_arm_streaks(image_hu, arm_band, body_mask, radius=5):
    """
    Arm band 내 streak artifact 픽셀만 주변 조직값으로 inpaint

    Streak = arm band 내에서 water-like HU (-30~30)인데
    주변 조직은 soft tissue인 곳 → 비정상적으로 낮은 HU = streak

    방법: streak 픽셀을 주변 non-streak 이웃의 weighted median으로 대체
    반복적으로 바깥에서 안쪽으로 채워나감 (iterative inpainting)
    """
    if not arm_band.any():
        return image_hu.copy()

    output = image_hu.copy().astype(np.float64)

    # Streak 판별: arm band 내 water-like HU
    # 주변 조직의 median HU를 구해서, 그보다 많이 낮은 픽셀 = streak
    band_pixels = arm_band & (body_mask > 0.5)

    # band 내 soft tissue 참조값 (streak이 아닌 정상 조직)
    non_water_in_band = band_pixels & ((image_hu < -30) | (image_hu > 30))
    if non_water_in_band.any():
        ref_hu = np.median(image_hu[non_water_in_band])
    else:
        ref_hu = 40.0  # soft tissue default

    # Streak mask: band 내 water-like HU 픽셀
    streak_mask = band_pixels & (image_hu >= -30) & (image_hu < 30)

    # 추가 조건: 주변과의 차이가 큰 것만 (진짜 streak vs 정상 경계)
    local_mean = ndimage.uniform_filter(image_hu.astype(np.float64), size=7)
    hu_deviation = np.abs(image_hu - local_mean)
    streak_mask = streak_mask & (hu_deviation > 10)

    n_streak = streak_mask.sum()
    print(f"    Arm band streak pixels: {n_streak}")

    if n_streak == 0:
        return output.astype(np.float32)

    # Iterative inpainting: 바깥 → 안쪽으로 점진적 채움
    # non-streak 이웃으로부터 값을 가져옴
    inpaint_result = output.copy()
    remaining = streak_mask.copy()

    for iteration in range(radius * 2):
        if not remaining.any():
            break

        # remaining의 경계 픽셀 = non-remaining과 인접한 remaining 픽셀
        dilated = ndimage.binary_dilation(~remaining, iterations=1)
        border = remaining & dilated

        if not border.any():
            # 고립된 픽셀이면 전체 처리
            border = remaining

        # border 픽셀마다 주변 non-remaining 이웃의 mean으로 대체
        # 효율적 구현: uniform_filter + count
        valid = (~remaining).astype(np.float64)
        value_sum = ndimage.uniform_filter(inpaint_result * valid, size=3) * 9
        count = ndimage.uniform_filter(valid, size=3) * 9

        # count > 0인 border 픽셀만 업데이트
        update_mask = border & (count > 0.5)
        if update_mask.any():
            inpaint_result[update_mask] = value_sum[update_mask] / count[update_mask]

        remaining[update_mask] = False

    # 아직 남은 픽셀은 참조값으로
    if remaining.any():
        inpaint_result[remaining] = ref_hu

    # streak 영역만 inpaint 결과로 교체
    output[streak_mask] = inpaint_result[streak_mask]

    return output.astype(np.float32)


def action_guided_denoise(image_hu, masks):
    """
    Action mask에 따른 적응적 denoising

    처리 로직:
    1. Arm band streak inpainting (블러링 아님, streak 픽셀만 주변값으로 보정)
    2. Wavelet denoise (baseline)
    3. Artifact (arm band 제외): median filter
    4. Structure mask 높은 영역: wavelet 약하게 (보존)
    5. Fluid 영역: 원본 보존
    6. noise_level에 따라 블렌딩 비율 조절
    """
    output = image_hu.copy().astype(np.float64)

    structure = masks['structure']
    artifact = masks['artifact']
    fluid = masks['fluid']
    noise_level = masks['noise_level']
    body = masks['body']
    arm_band = masks['arm_band']

    body_bool = body > 0.5

    # Step 0: Arm band streak inpainting (블러링 없이 streak만 보정)
    print("  Step 0: Arm band streak inpainting...")
    output = inpaint_arm_streaks(output, arm_band, body_bool).astype(np.float64)

    # Step 1: Wavelet denoise (전체)
    wavelet_result = wavelet_denoise(output.astype(np.float32), threshold_scale=2.5, level=2)

    # Step 2: Median filter (bone-proximity artifact용)
    median_3 = median_denoise(output.astype(np.float32), kernel_size=3)
    median_5 = median_denoise(output.astype(np.float32), kernel_size=5)

    # Step 3: 영역별 블렌딩

    # --- Artifact 영역 (arm band 제외): median filter ---
    # arm band는 이미 inpaint 했으므로, 나머지 artifact만 median 적용
    art_no_arm = artifact * (~arm_band).astype(np.float64)
    art_weight = np.clip(art_no_arm * 1.5, 0, 1)
    artifact_result = median_5 * 0.7 + wavelet_result * 0.3
    output = output * (1 - art_weight) + artifact_result * art_weight

    # --- Structure 보존 영역: wavelet만 약하게 ---
    struct_preserve = np.clip(structure * 1.2, 0, 1)
    struct_result = output * 0.8 + wavelet_result * 0.2
    non_artifact = (artifact < 0.15)
    blend_mask = struct_preserve * non_artifact.astype(np.float64)
    output = output * (1 - blend_mask) + struct_result * blend_mask

    # --- Noise level 기반 추가 처리 ---
    high_noise = body_bool & (noise_level > 1.3) & (artifact < 0.15)
    if high_noise.any():
        noise_blend = np.clip((noise_level - 1.0) / 1.5, 0, 0.5)
        noise_blend *= high_noise.astype(np.float64)
        noise_result = median_3 * 0.4 + wavelet_result * 0.6
        output = output * (1 - noise_blend) + noise_result * noise_blend

    # --- Fluid 영역: 원본 보존 ---
    fluid_weight = np.clip(fluid * 1.5, 0, 1)
    output = output * (1 - fluid_weight) + image_hu * fluid_weight

    # --- Body 외부: 건드리지 않음 ---
    output[~body_bool] = image_hu[~body_bool]

    return output.astype(np.float32)


# ============================================================
# Display utilities
# ============================================================

def normalize_hu_for_display(hu_array, window_center=40, window_width=400):
    """HU → [0,255] with Window/Level (Soft tissue: C=40, W=400)"""
    hu_min = window_center - window_width / 2
    hu_max = window_center + window_width / 2
    clipped = np.clip(hu_array, hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)


def orient(arr):
    """Raw → Display 회전보정 (왼쪽 90° + 상하반전)"""
    return np.flipud(np.rot90(arr, k=1))


# ============================================================
# Full Pipeline
# ============================================================

def process_slice(image_hu, output_dir=None, slice_label=""):
    """
    단일 slice 전체 파이프라인

    Returns:
        denoised_hu: denoised image in HU
        masks: action-guided masks dict
    """
    print(f"\n{'='*60}")
    print(f"Processing{' ' + slice_label if slice_label else ''}...")
    print(f"{'='*60}")
    print(f"  HU range: [{image_hu.min():.0f}, {image_hu.max():.0f}]")

    # Step 1: Action mask 생성
    masks, features = generate_action_masks(image_hu)

    # Step 2: Mask-guided denoising
    print("\nApplying Action-Guided Denoising...")
    denoised_hu = action_guided_denoise(image_hu, masks)

    # Metrics
    hu_lo, hu_hi = -1000, 1000
    orig_norm = np.clip((image_hu - hu_lo) / (hu_hi - hu_lo), 0, 1)
    dn_norm = np.clip((denoised_hu - hu_lo) / (hu_hi - hu_lo), 0, 1)
    ssim_val = ssim(orig_norm, dn_norm, data_range=1.0)

    body_bool = masks['body'] > 0.5
    if body_bool.any():
        orig_body_std = np.std(image_hu[body_bool])
        dn_body_std = np.std(denoised_hu[body_bool])
        noise_reduction = (1 - dn_body_std / orig_body_std) * 100
    else:
        orig_body_std = dn_body_std = noise_reduction = 0

    print(f"\n  Metrics:")
    print(f"    SSIM: {ssim_val:.4f}")
    print(f"    Body STD - Original: {orig_body_std:.2f}, Denoised: {dn_body_std:.2f}")
    print(f"    Noise reduction: {noise_reduction:.1f}%")

    # Save results if output_dir given
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _save_results(image_hu, denoised_hu, masks, ssim_val, output_dir, slice_label)

    return denoised_hu, masks


def _save_results(image_hu, denoised_hu, masks, ssim_val, output_dir, label=""):
    """결과 저장"""
    prefix = f"{label}_" if label else ""

    # CT images
    orig_disp = orient(normalize_hu_for_display(image_hu))
    dn_disp = orient(normalize_hu_for_display(denoised_hu))
    Image.fromarray(orig_disp).save(f"{output_dir}/{prefix}01_original.png")
    Image.fromarray(dn_disp).save(f"{output_dir}/{prefix}02_denoised.png")

    # Individual masks
    for name in ['structure', 'artifact', 'fluid', 'noise_level', 'body']:
        mask = masks[name]
        if name == 'noise_level':
            mask_norm = np.clip(mask / 3.0, 0, 1)
        else:
            mask_norm = np.clip(mask, 0, 1)
        mask_uint8 = orient((mask_norm * 255).astype(np.uint8))
        Image.fromarray(mask_uint8).save(f"{output_dir}/{prefix}mask_{name}.png")

    # Tissue label colored
    tissue_vis = np.zeros((*image_hu.shape, 3), dtype=np.uint8)
    for tid, color in TISSUE_COLORS.items():
        tissue_vis[masks['tissue_label'] == tid] = color
    # Save as PNG (oriented)
    tissue_oriented = orient(tissue_vis)
    Image.fromarray(tissue_oriented).save(f"{output_dir}/{prefix}mask_tissue.png")

    # Comparison figure
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))

    # Row 1: CT comparison
    axes[0, 0].imshow(orig_disp, cmap='gray')
    axes[0, 0].set_title('Original NC-CT', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(dn_disp, cmap='gray')
    axes[0, 1].set_title(f'Denoised (SSIM: {ssim_val:.4f})', fontsize=11)
    axes[0, 1].axis('off')

    diff = np.abs(orig_disp.astype(float) - dn_disp.astype(float))
    axes[0, 2].imshow(diff, cmap='hot', vmin=0, vmax=30)
    axes[0, 2].set_title('Difference (removed)', fontsize=11)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(orient(tissue_vis))
    axes[0, 3].set_title('Tissue Classification', fontsize=11)
    axes[0, 3].axis('off')

    # Row 2: Action masks
    axes[1, 0].imshow(orient(masks['structure']), cmap='RdYlGn', vmin=0, vmax=1)
    axes[1, 0].set_title('Structure (preserve)', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(orient(masks['artifact']), cmap='Reds', vmin=0, vmax=0.6)
    axes[1, 1].set_title(f'Artifact ({(masks["artifact"]>0.3).sum()}px)', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(orient(masks['fluid']), cmap='Blues', vmin=0, vmax=1)
    axes[1, 2].set_title(f'Fluid ({(masks["fluid"]>0.3).sum()}px)', fontsize=11)
    axes[1, 2].axis('off')

    axes[1, 3].imshow(orient(masks['noise_level']), cmap='hot', vmin=0, vmax=2)
    axes[1, 3].set_title('Noise Level', fontsize=11)
    axes[1, 3].axis('off')

    plt.suptitle(f'Action-Guided CT Denoising {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to: {output_dir}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # 설정
    patient_id = "1728852"
    slice_idx = 27
    nifti_path = f"F:/LD-CT SR/Data/NC-CT NIfTI/{patient_id}.nii"
    output_dir = "F:/LD-CT SR/Outputs/ma_hybrid/integrated_test"

    print("=" * 60)
    print("Action-Guided CT Denoising Pipeline")
    print("=" * 60)

    # NIfTI 로드
    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()
    image_hu = volume[:, :, slice_idx].astype(np.float32)

    print(f"Patient: {patient_id}")
    print(f"Volume: {volume.shape}")
    print(f"Slice: {slice_idx}")

    # 처리
    denoised, masks = process_slice(
        image_hu,
        output_dir=output_dir,
        slice_label=f"p{patient_id}_s{slice_idx}"
    )

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
