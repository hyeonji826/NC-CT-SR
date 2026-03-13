"""
3D Streak Detection for Low-Dose CT
=====================================
Sagittal multi-view streak intensity map 생성 (탐지만, 복원 X)

출력:
  streak_map: (H, W, Z) float32 - streak 강도 맵 (양=bright, 음=dark)
  anatomy_z:  (H, W, Z) float32 - z-median anatomy (pseudo-GT용)
  body_3d:    (H, W, Z) bool    - body mask

좌표계 (NIfTI raw -> display):
  displayed = raw.T  (rot90(k=-1) + fliplr)
  display-horizontal = raw axis 0 (y)
  display-vertical   = raw axis 1 (x)
  수평 streak: raw-y 방향으로 일정, raw-x 방향으로 변함
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import (median_filter, uniform_filter1d,
                            label, binary_fill_holes)


# ============================================================
# Volume I/O
# ============================================================

def load_volume(path):
    """NIfTI 로드 -> (H, W, Z) float32 HU"""
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    z_axis = int(np.argmin(vol.shape))
    if z_axis == 0:
        vol = np.transpose(vol, (1, 2, 0))
    elif z_axis == 1:
        vol = np.transpose(vol, (0, 2, 1))
    return vol, img.affine, img.header


# ============================================================
# Helpers
# ============================================================

def find_body_mask(slice_hu):
    """Body = largest connected component > -500 HU"""
    candidate = slice_hu > -500
    labeled, n = label(candidate)
    if n == 0:
        return np.zeros_like(slice_hu, dtype=bool)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return binary_fill_holes(labeled == np.argmax(sizes))


def detect_arm_band(slice_hu, body_mask, margin=15):
    """
    좌/우 팔뼈 사이 수평 밴드 검출
    Returns: dict {y_min, y_max, x_min, x_max} or None
    """
    H, W = slice_hu.shape
    bone = slice_hu > 200
    exterior_bone = bone & ~body_mask
    if not exterior_bone.any():
        return None

    body_ys = np.where(body_mask.any(axis=1))[0]
    if len(body_ys) == 0:
        return None
    body_cy = float(np.mean(body_ys))

    ext_ys, ext_xs = np.where(exterior_bone)
    left = ext_ys < body_cy
    right = ext_ys >= body_cy

    if left.sum() < 20 or right.sum() < 20:
        return None

    left_x_min, left_x_max = ext_xs[left].min(), ext_xs[left].max()
    right_x_min, right_x_max = ext_xs[right].min(), ext_xs[right].max()

    band_x_min = max(left_x_min, right_x_min) - margin
    band_x_max = min(left_x_max, right_x_max) + margin
    if band_x_min >= band_x_max:
        return None

    left_y = float(np.mean(ext_ys[left]))
    right_y = float(np.mean(ext_ys[right]))
    band_y_min = int(min(left_y, right_y)) - margin
    band_y_max = int(max(left_y, right_y)) + margin

    return {
        'y_min': max(0, band_y_min),
        'y_max': min(H, band_y_max),
        'x_min': max(0, band_x_min),
        'x_max': min(W, band_x_max),
    }


# ============================================================
# Edge-preserving Z-median
# ============================================================

def edge_preserving_z_median(volume, kernel_z=5, edge_hu=50, verbose_fn=None):
    """
    Z-방향 median 필터 + 조직 경계 보존.

    각 voxel에 대해 z-neighborhood (kernel_z)를 보되,
    center와 |HU 차이| > edge_hu인 neighbor는 제외하고 median 계산.

    fat(-80) ↔ soft tissue(+40) = ~120 HU 차이 → 제외됨 (경계 보존)
    같은 조직 내 noise ~20-30 HU → 포함됨 (평활화 유지)

    Args:
        volume: (H, W, Z) float32
        kernel_z: z-direction kernel size (odd)
        edge_hu: HU threshold for edge detection
    """
    pr = verbose_fn or (lambda *a, **k: None)
    H, W, Z = volume.shape
    half_k = kernel_z // 2

    # Pad z-axis with edge replication
    padded = np.pad(volume, ((0, 0), (0, 0), (half_k, half_k)), mode='edge')

    # Build neighbor stack: (H, W, Z, kernel_z)
    neighbors = np.stack(
        [padded[:, :, i:i + Z] for i in range(kernel_z)],
        axis=-1
    )
    center = volume[:, :, :, np.newaxis]  # (H, W, Z, 1)

    # Mask: only neighbors within edge_hu of center
    close = np.abs(neighbors - center) <= edge_hu  # (H, W, Z, kernel_z)

    # Center slice (index half_k) is always included (diff=0)
    n_excluded = (~close).sum()
    n_total = close.size
    pr(f"  Edge-preserving: {n_excluded}/{n_total} neighbor-voxels excluded "
       f"({100.*n_excluded/n_total:.1f}%)")

    # Masked median via nanmedian
    masked = np.where(close, neighbors, np.nan)
    with np.errstate(all='ignore'):
        anatomy = np.nanmedian(masked, axis=-1).astype(np.float32)

    # Fallback (shouldn't happen: center always close to itself)
    nan_mask = np.isnan(anatomy)
    if nan_mask.any():
        anatomy[nan_mask] = volume[nan_mask]

    return anatomy


# ============================================================
# Sagittal-domain streak estimation
# ============================================================

def estimate_streak_sagittal(residual, body_3d, smooth_x=3):
    """
    Sagittal row-mean -> streak profile (W, Z)
    수평 streak은 raw-y로 일정 -> y-mean이 streak common-mode 추출
    """
    masked = residual.copy()
    masked[~body_3d] = 0.0
    streak_sum = masked.sum(axis=0)
    count = body_3d.sum(axis=0).astype(np.float32)

    valid = count > 5
    streak_xz = np.zeros_like(streak_sum, dtype=np.float32)
    streak_xz[valid] = (streak_sum[valid] / count[valid]).astype(np.float32)

    if smooth_x > 1:
        for z in range(residual.shape[2]):
            streak_xz[:, z] = uniform_filter1d(
                streak_xz[:, z].astype(np.float64), size=smooth_x
            ).astype(np.float32)

    return streak_xz


def estimate_streak_profile(volume, body_3d, smooth_w=101, verbose_fn=None):
    """
    Per-slice H-mean profile 기반 streak 추정.
    z-median에 의존하지 않아 z-방향으로 일정한 streak도 탐지 가능.

    각 (W, Z)에서:
      1. H축 soft tissue body pixel의 평균 HU → raw profile
      2. W축 count-weighted smoothing (큰 커널) → anatomy baseline
      3. raw - baseline = streak estimate

    큰 smooth_w로 anatomy의 넓은 구조를 추정하고,
    streak의 좁은 band (~10-40px)는 잔차로 남김.

    Args:
        volume: (H, W, Z) float32
        body_3d: (H, W, Z) bool
        smooth_w: W축 smoothing kernel 크기 (anatomy scale)
    Returns:
        streak_prof: (W, Z) float32
    """
    pr = verbose_fn or (lambda *a, **k: None)
    H, W, Z = volume.shape

    # Soft tissue body mask (bone 제외, air 제외)
    soft = body_3d & (volume > -500) & (volume < 200)

    # H축 합계와 카운트: (W, Z)
    masked_vol = np.where(soft, volume, 0.0)
    h_sum = masked_vol.sum(axis=0).astype(np.float64)    # (W, Z)
    h_count = soft.sum(axis=0).astype(np.float64)         # (W, Z)

    # Raw H-mean profile (충분한 H-count가 있는 위치만)
    min_count = max(20, int(0.04 * H))
    valid = h_count >= min_count
    profile = np.zeros((W, Z), dtype=np.float64)
    profile[valid] = h_sum[valid] / h_count[valid]
    # Invalid 위치는 0 → count-weighted smoothing에서 자동 처리

    # Count-weighted smoothing along W (axis=0)
    # baseline = smooth(sum) / smooth(count)
    # → body edge에서 count가 적으면 자연히 nearby 값으로 보간
    sum_smooth = uniform_filter1d(h_sum, size=smooth_w, axis=0)
    count_smooth = uniform_filter1d(h_count, size=smooth_w, axis=0)

    baseline = np.zeros((W, Z), dtype=np.float64)
    valid_smooth = count_smooth >= min_count
    baseline[valid_smooth] = sum_smooth[valid_smooth] / count_smooth[valid_smooth]

    # Streak = raw profile - smoothed baseline
    streak_prof = np.zeros((W, Z), dtype=np.float32)
    both = valid & valid_smooth
    streak_prof[both] = (profile[both] - baseline[both]).astype(np.float32)

    # Body edge clamp: H-count가 적은 위치는 streak 값 축소
    # (H-count가 min_count 근처이면 신뢰도 낮음)
    count_ratio = np.zeros((W, Z), dtype=np.float32)
    max_count = h_count.max()
    if max_count > 0:
        count_ratio[both] = (h_count[both] / max_count).astype(np.float32)
    # count_ratio < 0.2이면 edge → 값 감쇠
    edge_weight = np.clip(count_ratio / 0.2, 0, 1)
    streak_prof *= edge_weight

    pr(f"  Profile-based: [{streak_prof.min():.1f}, {streak_prof.max():.1f}] HU "
       f"(smooth_w={smooth_w})")
    return streak_prof


# ============================================================
# Main: Streak Detection Pipeline
# ============================================================

def detect_streaks_3d(volume, kernel_z=5, edge_hu=50,
                      exposure_mAs=None, verbose=True):
    """
    3D Streak Detection - Sagittal Multi-View

    Pipeline:
      1. Edge-preserving z-median -> anatomy estimate (경계 보존)
      2. Sagittal row-mean of z-residual -> global streak profile
      3. Per-slice H-mean profile -> arm-band streak (z-일정 streak 탐지)

    Args:
        volume: (H, W, Z) float32 HU
        kernel_z: z-median kernel (odd)
        edge_hu: z-neighbor 제외 threshold (HU). fat↔tissue 경계 보존.
        exposure_mAs: DICOM Exposure (mAs)

    Returns:
        streak_map: (H, W, Z) float32 - streak 강도 (양=bright, 음=dark)
        anatomy_z:  (H, W, Z) float32 - edge-preserving z-median anatomy
        body_3d:    (H, W, Z) bool    - body mask
    """
    pr = print if verbose else (lambda *a, **k: None)
    H, W, Z = volume.shape

    REF_MAS = 28.0
    if exposure_mAs is not None and exposure_mAs > 0:
        noise_factor = np.sqrt(REF_MAS / exposure_mAs)
    else:
        noise_factor = 1.0

    pr(f"=== 3D Streak Detection (sagittal multi-view) ===")
    pr(f"Volume: {H}x{W}x{Z}, HU: [{volume.min():.0f}, {volume.max():.0f}]")
    if exposure_mAs is not None:
        pr(f"Exposure: {exposure_mAs} mAs -> noise_factor={noise_factor:.2f}")

    # Step 1a: Simple z-median for streak detection (aggressive smoothing)
    pr(f"Step 1a: Simple z-median for streak detection (kernel={kernel_z})...")
    anatomy_smooth = median_filter(volume, size=(1, 1, kernel_z)).astype(np.float32)
    residual = volume - anatomy_smooth
    pr(f"  Residual mean |r|: {np.abs(residual).mean():.1f} HU")

    # Step 1b: Edge-preserving z-median for output anatomy (경계 보존)
    pr(f"Step 1b: Edge-preserving z-median for anatomy (edge_hu={edge_hu})...")
    anatomy_z = edge_preserving_z_median(volume, kernel_z, edge_hu, verbose_fn=pr)

    # Step 2: Body masks
    pr("Step 2: Body masks...")
    body_3d = np.zeros_like(volume, dtype=bool)
    for z in range(Z):
        body_3d[:, :, z] = find_body_mask(volume[:, :, z])

    # Step 3: Sagittal-domain streak estimation (global)
    pr("Step 3: Sagittal-domain streak estimation...")
    streak_xz = estimate_streak_sagittal(residual, body_3d, smooth_x=3)
    pr(f"  Global profile: [{streak_xz.min():.1f}, {streak_xz.max():.1f}] HU")

    # Broadcast to 3D
    streak_map = np.broadcast_to(
        streak_xz[np.newaxis, :, :], (H, W, Z)
    ).copy()
    streak_map[~body_3d] = 0

    # Step 3b: Profile-based streak estimation for arm band
    pr("Step 3b: Profile-based arm-band streak estimation...")
    streak_prof = estimate_streak_profile(volume, body_3d, smooth_w=101,
                                          verbose_fn=pr)

    # Arm band detection per slice + profile 값 적용
    n_band = 0
    for z in range(Z):
        sl = volume[:, :, z]
        body = body_3d[:, :, z]
        band = detect_arm_band(sl, body)
        if band is None:
            continue
        n_band += 1
        x0, x1 = band['x_min'], band['x_max']

        # Arm band W 범위에서: z-median과 profile 중 더 강한 신호 사용
        for x in range(x0, x1):
            prof_val = streak_prof[x, z]
            zmed_val = streak_xz[x, z]
            if abs(prof_val) > abs(zmed_val):
                streak_map[:, x, z] = np.where(
                    body_3d[:, x, z], prof_val, 0.0
                )
    pr(f"  Arm band: {n_band}/{Z} slices")
    pr(f"  Profile replaced stronger in arm band")

    # Body 밖 제거
    streak_map[~body_3d] = 0

    # Stats
    body_stk = streak_map[body_3d]
    pr(f"\nStreak map stats (body only):")
    pr(f"  Range: [{body_stk.min():.1f}, {body_stk.max():.1f}] HU")
    pr(f"  |streak| > 5 HU:  {(np.abs(body_stk) > 5).sum()} voxels")
    pr(f"  |streak| > 10 HU: {(np.abs(body_stk) > 10).sum()} voxels")
    pr(f"  |streak| > 30 HU: {(np.abs(body_stk) > 30).sum()} voxels")

    return streak_map, anatomy_z, body_3d
