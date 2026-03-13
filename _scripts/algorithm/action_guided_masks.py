"""
Action-Guided Mask System v3 (Enhanced)
========================================
v2 대비 개선:
1. 폐/공기 영역 artifact 억제: 상부 슬라이스에서 늑골 근처 폐를 artifact로 오분류 방지
2. Between-bones 탐지 범위 확대: 50→80px (골반부 장골 사이 streak 커버)
3. Arm band 정밀화: base 확률 제거, gradient/deviation 증거가 있을 때만 artifact
4. Tissue-aware artifact: 연조직/지방 HU 범위만 artifact 후보 (공기/폐 제외)

생성하는 마스크:
1. structure_mask: 보존해야 할 해부학적 구조 (0-1, continuous)
2. artifact_mask: 제거해야 할 streak artifact (0-1, continuous)
3. fluid_mask: 보존해야 할 유체 영역 (0-1, continuous)
4. noise_level_map: 각 픽셀의 추정 노이즈 레벨 (denoise 강도 가이드)
5. tissue_label_map: 기본 조직 분류 (background/fat/soft/bone)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter, label, binary_fill_holes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os


# ============================================================
# Tissue labels (from v2 classifier)
# ============================================================
TISSUE_BG = 0
TISSUE_FAT = 1
TISSUE_SOFT = 2
TISSUE_BONE = 3
TISSUE_FLUID = 4
TISSUE_VESSEL = 5
TISSUE_ARTIFACT = 6

TISSUE_NAMES = {
    0: 'background', 1: 'fat', 2: 'soft_tissue',
    3: 'bone', 4: 'fluid', 5: 'vessel', 6: 'artifact',
}
TISSUE_COLORS = {
    0: [0, 0, 0],        # background: black
    1: [255, 200, 50],    # fat: yellow
    2: [255, 100, 100],   # soft tissue: red
    3: [150, 150, 255],   # bone: blue
    4: [50, 200, 255],    # fluid: cyan
    5: [255, 50, 200],    # vessel: magenta
    6: [200, 50, 50],     # artifact: dark red
}


# ============================================================
# Feature computation
# ============================================================

def detect_arm_artifact_band(image_hu, body_mask, margin=30):
    """
    팔뼈 사이 수평 artifact band 검출

    v3: arm_band (bool) + arm_dist_weight (float, 0~1) 반환
    - arm_dist_weight: 팔뼈에 가까울수록 1, 멀수록(척추 근처) 감쇠
    - 팔뼈→척추까지는 artifact, 척추 너머는 proximity가 처리

    Returns:
        arm_band: bool mask (H,W)
        arm_dist_weight: float32 (H,W) 0~1 - 팔뼈 근접 가중치
    """
    H, W = image_hu.shape
    body_ys, body_xs = np.where(body_mask)
    body_cy = np.mean(body_ys)

    # Body 외부 bone = arm bone
    bone_mask = image_hu > 200
    exterior_bone = bone_mask & ~body_mask
    if not exterior_bone.any():
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    labeled_ext, n_ext = label(exterior_bone)
    ext_sizes = np.bincount(labeled_ext.ravel())
    ext_sizes[0] = 0

    arm_components = []
    for i in range(1, len(ext_sizes)):
        if ext_sizes[i] < 20:
            continue
        coords = np.where(labeled_ext == i)
        cy = np.mean(coords[0])
        arm_components.append({
            'raw_cy': cy,
            'raw_x_min': coords[1].min(), 'raw_x_max': coords[1].max(),
        })

    # 좌/우 분리
    left_arms = [ac for ac in arm_components if ac['raw_cy'] < body_cy]
    right_arms = [ac for ac in arm_components if ac['raw_cy'] >= body_cy]

    if not left_arms or not right_arms:
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    # 양쪽 arm의 raw_x 겹침 구간 = artifact band
    left_x_min = min(ac['raw_x_min'] for ac in left_arms)
    left_x_max = max(ac['raw_x_max'] for ac in left_arms)
    right_x_min = min(ac['raw_x_min'] for ac in right_arms)
    right_x_max = max(ac['raw_x_max'] for ac in right_arms)

    band_x_min = max(0, max(left_x_min, right_x_min) - margin)
    band_x_max = min(W, min(left_x_max, right_x_max) + margin)

    # raw_y 범위: 좌팔 ~ 우팔
    left_y_center = np.mean([ac['raw_cy'] for ac in left_arms])
    right_y_center = np.mean([ac['raw_cy'] for ac in right_arms])
    band_y_min = max(0, int(left_y_center) - 10)
    band_y_max = min(H, int(right_y_center) + 10)

    arm_band = np.zeros((H, W), dtype=bool)
    arm_band[band_y_min:band_y_max, band_x_min:band_x_max] = True
    arm_band &= body_mask

    # === v3: 팔뼈로부터의 거리 기반 가중치 ===
    # 팔뼈(exterior bone)에 가까울수록 1, 멀수록 0으로 감쇠
    # → 팔뼈~척추까지는 높은 가중치, 척추 너머는 자연스럽게 감소
    dist_from_arm = ndimage.distance_transform_edt(~exterior_bone).astype(np.float32)

    # 척추까지의 대략적 거리: body 폭의 ~40% (팔에서 중심까지)
    body_width = body_xs.max() - body_xs.min() if len(body_xs) > 0 else W
    decay_dist = body_width * 0.45  # 팔뼈~척추 거리 추정

    arm_dist_weight = np.clip(1.0 - dist_from_arm / max(decay_dist, 1), 0, 1)
    arm_dist_weight *= arm_band.astype(np.float32)  # band 내에서만 유효

    return arm_band, arm_dist_weight


def detect_photon_starvation(image_hu, body_mask, arm_band):
    """
    Photon starvation 검출: 팔 사이에서 X선 감쇠로 인한 어두운 밴드.

    원리:
    - Arm band 밖의 정상 body 조직으로부터 기대 HU 추정 (large-scale smoothing)
    - Band 내에서 기대보다 유의미하게 어두운 영역 = photon starvation
    - 감쇠된 연조직: 정상 ~40 HU → starvation 시 -50~10 HU

    Returns:
        starvation_mask: bool (H, W)
        starvation_weight: float32 (H, W) 0~1 - 감쇠 강도
    """
    H, W = image_hu.shape

    if not arm_band.any():
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    # Band 밖 body 조직에서 기대 HU 추정
    non_band_body = body_mask & ~arm_band

    if non_band_body.sum() < 100:
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    weight = non_band_body.astype(np.float64)
    value = image_hu.astype(np.float64) * weight

    # 점진적으로 큰 커널 사용 → band 중심까지 기대값 외삽
    expected_hu = np.full((H, W), np.nan, dtype=np.float64)

    for kernel in [51, 81, 121]:
        smooth_val = uniform_filter(value, size=kernel) * (kernel ** 2)
        smooth_wt = uniform_filter(weight, size=kernel) * (kernel ** 2)

        new_valid = (smooth_wt > 3) & np.isnan(expected_hu)
        if new_valid.any():
            expected_hu[new_valid] = smooth_val[new_valid] / smooth_wt[new_valid]

    # NaN 잔여: nearest valid value로 채우기
    nan_mask = np.isnan(expected_hu)
    if nan_mask.any() and (~nan_mask).any():
        _, nearest_idx = ndimage.distance_transform_edt(nan_mask, return_indices=True)
        expected_hu = expected_hu[tuple(nearest_idx)]
    elif nan_mask.all():
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    # HU drop: 양수 = 기대보다 어두움
    hu_drop = expected_hu - image_hu

    # Starvation 판정:
    # - arm band 내, body 내
    # - 기대치보다 25+ HU 어두움
    # - bone/extreme background 아님
    starvation = (
        arm_band & body_mask &
        (hu_drop > 25) &
        (image_hu > -300) & (image_hu < 200)
    )

    # 너무 작은 영역 제거 (노이즈)
    labeled_s, n_s = label(starvation)
    if n_s > 0:
        sizes_s = np.bincount(labeled_s.ravel())
        sizes_s[0] = 0
        for i in range(1, len(sizes_s)):
            if sizes_s[i] < 50:
                starvation[labeled_s == i] = False

    # 연속 가중치 (drop 크기에 비례)
    starvation_weight = np.clip(hu_drop / 80, 0, 1).astype(np.float32)
    starvation_weight *= starvation.astype(np.float32)

    return starvation, starvation_weight


def inpaint_photon_starvation(image_hu, starvation_mask, body_mask, arm_band=None):
    """
    Photon starvation 영역을 expected HU fill + boundary blending으로 복원.

    원리:
    - Band 밖 정상 조직에서 기대 HU 추정 (large-scale smoothing)
    - Starvation 영역을 기대값으로 채움 (방향성 없음 → 조직 경계 안 넘음)
    - 경계부는 iterative blending으로 자연스러운 전이

    Returns:
        corrected: float32 (H, W)
        n_fixed: 보정된 픽셀 수
    """
    H, W = image_hu.shape
    n_fixed = int(starvation_mask.sum())
    if n_fixed == 0:
        return image_hu.copy(), 0

    output = image_hu.copy().astype(np.float64)

    # 1. Expected HU 계산: band 밖 body 조직 기반 large-scale smoothing
    if arm_band is None:
        arm_band = starvation_mask  # fallback
    non_affected = body_mask & ~starvation_mask
    weight = non_affected.astype(np.float64)
    value = image_hu.astype(np.float64) * weight

    expected_hu = np.full((H, W), np.nan, dtype=np.float64)
    for kernel in [31, 51, 81, 121]:
        smooth_val = uniform_filter(value, size=kernel) * (kernel ** 2)
        smooth_wt = uniform_filter(weight, size=kernel) * (kernel ** 2)
        new_valid = (smooth_wt > 3) & np.isnan(expected_hu)
        if new_valid.any():
            expected_hu[new_valid] = smooth_val[new_valid] / smooth_wt[new_valid]

    # NaN 잔여 채우기
    nan_mask = np.isnan(expected_hu)
    if nan_mask.any() and (~nan_mask).any():
        _, nearest_idx = ndimage.distance_transform_edt(nan_mask, return_indices=True)
        expected_hu = expected_hu[tuple(nearest_idx)]
    elif nan_mask.all():
        return image_hu.copy(), 0

    # 2. Starvation 영역을 expected HU로 채움
    output[starvation_mask] = expected_hu[starvation_mask]

    # 3. Boundary blending: 경계부를 iterative smoothing으로 자연스럽게
    # starvation 경계 3px 확장 → smoothing 적용
    dilated = ndimage.binary_dilation(starvation_mask, iterations=3)
    blend_zone = dilated & ~starvation_mask & body_mask

    for _ in range(5):
        smoothed = uniform_filter(output, size=5)
        # blend_zone에서만 부분 적용 (0.3 blend)
        output[blend_zone] = output[blend_zone] * 0.7 + smoothed[blend_zone] * 0.3

    return output.astype(np.float32), n_fixed


def compute_between_bones(image_hu, max_dist=80):
    """
    각 픽셀이 뼈 쌍 사이에 있는지 검출 (directional kernel 방식 - 빠름)

    원리: 4개 축 방향(0°, 45°, 90°, 135°)에서
    forward/backward 양쪽에 bone이 있으면 "between bones"

    v3: max_dist 50→80 (골반부 장골 사이 거리 커버)

    Returns:
        between_bones_count: 몇 개 축에서 양쪽 bone이 있는지 (0~4)
    """
    H, W = image_hu.shape
    bone_mask = (image_hu > 200).astype(np.float32)

    if not bone_mask.any():
        return np.zeros((H, W), dtype=np.int32)

    # 4개 축 방향 정의
    directions = [
        (0, 1),   # Horizontal (0°)
        (1, 1),   # Diagonal 45°
        (1, 0),   # Vertical (90°)
        (1, -1),  # Diagonal 135°
    ]

    between_count = np.zeros((H, W), dtype=np.int32)

    for dy, dx in directions:
        # Forward 방향에 bone이 있는지: bone_mask를 해당 방향으로 dilate
        # Efficient: 1D maximum filter along the direction

        # Forward: shift bone_mask in -direction (so bone "spreads" forward)
        found_forward = np.zeros((H, W), dtype=bool)
        found_backward = np.zeros((H, W), dtype=bool)

        # Accumulate: 각 step에서 shifted bone mask를 OR
        for step in range(5, max_dist + 1, 3):  # 5부터 시작 (bone 자체 제외), 3px 간격으로 빠르게
            sy, sx = dy * step, dx * step

            # Forward
            if abs(sy) < H and abs(sx) < W:
                shifted = ndimage.shift(bone_mask, [-sy, -sx], order=0, mode='constant', cval=0)
                found_forward |= shifted > 0.5

            # Backward
            if abs(sy) < H and abs(sx) < W:
                shifted = ndimage.shift(bone_mask, [sy, sx], order=0, mode='constant', cval=0)
                found_backward |= shifted > 0.5

        # 양쪽에 bone → between bones
        between_count += (found_forward & found_backward).astype(np.int32)

    return between_count


def compute_directional_anisotropy(hu_f64, kernel_size=15):
    """
    방향별 variance 비율로 streak artifact의 선형 패턴 검출.

    원리:
    - Streak artifact: 특정 방향(주로 수평)으로 일관된 줄무늬 → 한 축 variance↑, 수직 축↓
    - 등방성 noise/구조: 모든 방향에서 유사한 variance
    - Fluid: 모든 방향에서 낮은 variance

    1D uniform_filter를 수평/수직으로 적용하여 방향별 variance를 계산하고,
    max(h,v) / min(h,v) 비율로 anisotropy를 0~1로 정규화.

    Returns:
        anisotropy: float32 (H,W) 0~1. 1 = 강한 방향성(streak), 0 = 등방성
    """
    # Horizontal variance (수평 1D 커널 → 세로 streak 감지)
    h_kernel = np.ones((1, kernel_size), dtype=np.float64) / kernel_size
    h_mean = ndimage.convolve(hu_f64, h_kernel, mode='reflect')
    h_mean_sq = ndimage.convolve(hu_f64 ** 2, h_kernel, mode='reflect')
    h_var = np.maximum(h_mean_sq - h_mean ** 2, 0)

    # Vertical variance (수직 1D 커널 → 가로 streak 감지)
    v_kernel = np.ones((kernel_size, 1), dtype=np.float64) / kernel_size
    v_mean = ndimage.convolve(hu_f64, v_kernel, mode='reflect')
    v_mean_sq = ndimage.convolve(hu_f64 ** 2, v_kernel, mode='reflect')
    v_var = np.maximum(v_mean_sq - v_mean ** 2, 0)

    # Anisotropy: max/min ratio, 정규화
    max_var = np.maximum(h_var, v_var)
    min_var = np.minimum(h_var, v_var) + 1e-6
    # ratio=1 → 등방성, ratio>1 → 방향성. (ratio-1)/3으로 0~1 매핑
    anisotropy = np.clip((max_var / min_var - 1.0) / 3.0, 0, 1)

    return anisotropy.astype(np.float32)


def compute_local_kurtosis(hu_f64, kernel_size=9):
    """
    Local excess kurtosis (4차 모멘트).

    물리적 의미:
    - Fat/Fluid: Gaussian-like 분포 → excess kurtosis ≈ 0
    - Artifact: streak의 밝/어 교대로 heavy-tail → excess kurtosis > 2
    - Soft tissue: 정상 조직 → kurtosis ≈ 0~1

    수학: kurtosis = E[(X-μ)⁴] / E[(X-μ)²]² - 3
    """
    mean = uniform_filter(hu_f64, size=kernel_size)
    diff = hu_f64 - mean
    m2 = uniform_filter(diff ** 2, size=kernel_size)
    m4 = uniform_filter(diff ** 4, size=kernel_size)

    # Excess kurtosis (정상분포=0), 분산 너무 작으면 0으로 처리
    kurtosis = np.where(m2 > 1.0, m4 / (m2 ** 2 + 1e-8) - 3.0, 0.0)
    return np.clip(kurtosis, -2, 20).astype(np.float32)


def compute_local_range(hu_f64, kernel_size=9):
    """
    Local HU range (max - min within window).

    물리적 의미:
    - Fluid: 매우 균일 → range < 30 HU
    - Fat: 균일 → range < 60 HU
    - Soft tissue: 중간 → range 50~150 HU
    - Artifact: streak으로 HU 진폭 큼 → range > 100 HU

    std와의 차이: std는 평균 편차, range는 극단값 차이
    → artifact의 bright/dark streak band를 range가 더 민감하게 잡음
    """
    from scipy.ndimage import maximum_filter, minimum_filter
    local_max = maximum_filter(hu_f64, size=kernel_size)
    local_min = minimum_filter(hu_f64, size=kernel_size)
    return (local_max - local_min).astype(np.float32)


def compute_vesselness(hu_f64, sigma=3.0):
    """
    Hessian 고유값 기반 vesselness (Frangi filter 단순화).

    물리적 의미:
    - Vessel: 단면이 원형/타원형 관 → 한 방향 고유값 크고 수직 방향 작음
    - Fluid pool: 대칭적 blob → 양쪽 고유값 비슷
    - Fat: 균일 → 양쪽 고유값 모두 작음

    수학: Hessian H의 eigenvalue λ_s, λ_l (|λ_s| ≤ |λ_l|)
    R_B = |λ_s|/|λ_l| → 0=관형, 1=blob
    S = √(λ_s² + λ_l²) → 구조 강도
    Vesselness = exp(-R²/2β²) × (1 - exp(-S²/2c²))

    sigma=3.0: ~2.3mm scale → 소혈관(3-5mm) 이상 검출, 노이즈 무시
    """
    from scipy.ndimage import gaussian_filter

    # Hessian matrix: 2차 편미분 (Gaussian 스무딩 후)
    Hxx = gaussian_filter(hu_f64, sigma=sigma, order=[0, 2])
    Hyy = gaussian_filter(hu_f64, sigma=sigma, order=[2, 0])
    Hxy = gaussian_filter(hu_f64, sigma=sigma, order=[1, 1])

    # Eigenvalues of 2x2 Hessian
    trace = Hxx + Hyy
    det = Hxx * Hyy - Hxy ** 2
    discriminant = np.maximum(trace ** 2 - 4 * det, 0)
    sqrt_disc = np.sqrt(discriminant)

    lam1 = (trace - sqrt_disc) / 2
    lam2 = (trace + sqrt_disc) / 2

    # Sort by absolute value: |λ_s| ≤ |λ_l|
    abs1, abs2 = np.abs(lam1), np.abs(lam2)
    lam_s = np.where(abs1 <= abs2, lam1, lam2)
    lam_l = np.where(abs1 <= abs2, lam2, lam1)

    abs_s = np.abs(lam_s)
    abs_l = np.abs(lam_l)

    # Frangi vesselness
    beta = 0.5
    R_B = abs_s / (abs_l + 1e-8)
    S = np.sqrt(lam_s ** 2 + lam_l ** 2)

    # c = S의 75th percentile (선별적: 상위 25%만 높은 vesselness 부여)
    S_positive = S[S > 0]
    c = np.percentile(S_positive, 75) if len(S_positive) > 100 else 1.0

    vesselness = np.exp(-R_B ** 2 / (2 * beta ** 2)) * \
                 (1 - np.exp(-S ** 2 / (2 * c ** 2 + 1e-8)))

    # 구조 강도 없으면 vesselness 0
    vesselness[S < 1e-6] = 0

    return np.clip(vesselness, 0, 1).astype(np.float32)


def compute_local_connectivity(image_hu, body_mask):
    """
    Fat HU 범위 픽셀의 local connectivity (morphological closing).

    원리:
    - Real fat: 큰 연결 blob (피하지방층, 내장지방)
    - Artifact: 고립된 streak/반점 (photon starvation으로 HU가 fat 범위로 떨어진 조직)

    방법:
    Binary morphological closing으로 작은 gap 채움 → connected component 크기 측정
    Fat blob은 closing 후에도 크게 유지, artifact는 작은 조각들

    Returns:
        connectivity: float32 (H,W) 0~1. 1=큰 연결 blob(진짜 fat), 0=고립(artifact)
    """
    from scipy.ndimage import binary_closing

    # Fat HU 후보 (-120 ~ -30)
    fat_candidate = body_mask & (image_hu > -120) & (image_hu < -30)

    if not fat_candidate.any():
        return np.zeros_like(image_hu, dtype=np.float32)

    # Morphological closing: 5x5 구조요소, 2회 반복 → 작은 gap 채움
    struct = np.ones((5, 5), dtype=bool)
    closed = binary_closing(fat_candidate, structure=struct, iterations=2)

    # Connected component labeling
    labeled_c, n_comp = label(closed)

    connectivity = np.zeros_like(image_hu, dtype=np.float32)
    if n_comp > 0:
        sizes = np.bincount(labeled_c.ravel())
        sizes[0] = 0
        # 크기 기반 score: 200px 이상 → 1.0 (진짜 fat blob)
        for i in range(1, len(sizes)):
            score = np.clip(sizes[i] / 200.0, 0, 1)
            connectivity[labeled_c == i] = score

    # fat_candidate 밖은 0으로 유지
    connectivity *= fat_candidate.astype(np.float32)

    return connectivity


def compute_action_features(image_hu):
    """Action mask에 필요한 모든 feature 계산"""
    hu_f64 = image_hu.astype(np.float64)

    features = {}

    # Multi-scale local std
    for size in [3, 7, 15, 21]:
        mean = uniform_filter(hu_f64, size=size)
        mean_sq = uniform_filter(hu_f64**2, size=size)
        var = np.maximum(mean_sq - mean**2, 0)
        features[f'std_{size}'] = np.sqrt(var).astype(np.float32)

    # Local mean (smoothed HU)
    features['local_mean'] = uniform_filter(hu_f64, size=5).astype(np.float32)
    features['local_mean_21'] = uniform_filter(hu_f64, size=21).astype(np.float32)

    # HU deviation from large-scale context
    features['hu_deviation'] = np.abs(image_hu - features['local_mean_21'])

    # Gradient
    grad_x = ndimage.sobel(hu_f64, axis=1)
    grad_y = ndimage.sobel(hu_f64, axis=0)
    features['grad_mag'] = np.sqrt(grad_x**2 + grad_y**2).astype(np.float32)

    # Gradient direction consistency
    grad_dir = np.arctan2(grad_y, grad_x)
    mean_cos = uniform_filter(np.cos(grad_dir), size=7)
    mean_sin = uniform_filter(np.sin(grad_dir), size=7)
    features['dir_consistency'] = np.sqrt(mean_cos**2 + mean_sin**2).astype(np.float32)

    # Distance from background
    bg_mask = image_hu < -500
    features['dist_from_bg'] = ndimage.distance_transform_edt(~bg_mask).astype(np.float32)

    # Distance from bone
    bone_mask = image_hu > 200
    if bone_mask.any():
        features['dist_from_bone'] = ndimage.distance_transform_edt(~bone_mask).astype(np.float32)
    else:
        features['dist_from_bone'] = np.full_like(image_hu, 999, dtype=np.float32)

    # Between-bones detection (streak artifact의 물리적 원인)
    # v3: max_dist 50→80 (골반부 장골 커버)
    features['between_bones'] = compute_between_bones(image_hu, max_dist=80)

    # Directional anisotropy: streak artifact의 방향성 검출
    features['anisotropy'] = compute_directional_anisotropy(hu_f64)

    # Local kurtosis (4차 모멘트): 분포 꼬리 두께
    # artifact → heavy tail (kurtosis↑), 정상조직 → Gaussian-like (kurtosis≈0)
    features['kurtosis'] = compute_local_kurtosis(hu_f64)

    # Local range (max-min): 극값 진폭
    # fluid/fat(작음) vs artifact(큼) — std와 독립적으로 극단값 포착
    features['local_range'] = compute_local_range(hu_f64)

    # Vesselness (Hessian 고유값): 관형 구조 검출
    # vessel(관형, vesselness↑) vs fluid pool(blob, vesselness↓)
    features['vesselness'] = compute_vesselness(hu_f64)

    # Arm artifact band는 body_mask 필요하므로 나중에 추가됨 (generate_action_masks에서)

    return features


# ============================================================
# Body detection
# ============================================================

def find_body_mask(image_hu):
    """Body region 검출"""
    body_candidate = image_hu > -500
    labeled, n_features = label(body_candidate)
    if n_features == 0:
        return np.zeros_like(image_hu, dtype=bool)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = np.argmax(sizes)
    body_mask = labeled == largest
    body_mask = binary_fill_holes(body_mask)
    return body_mask


# ============================================================
# 1. Tissue classification (multi-feature, 6-class)
# ============================================================

def classify_tissue(image_hu, features, body_mask):
    """
    Multi-feature 조직 분류 (7 class)

    7개 독립 특성축으로 각 조직의 confidence score를 계산:
      축1. HU (local_mean)    — 밀도 기반 기본 구분
      축2. Std (std_7)        — 텍스처 분산 크기
      축3. Anisotropy         — 분산의 방향성 (등방 vs 이방)
      축4. Kurtosis           — 분포 꼬리 형태 (Gaussian vs heavy-tail)
      축5. Local Range        — 극값 HU 진폭
      축6. Vesselness         — Hessian 고유값 기반 관형 구조
      축7. Connectivity       — morphological closing 후 blob 크기 (fat vs artifact)

    Spatial context:
      arm_band / starvation_mask 영역에서는 fat 분류 차단.
      artifact_zone 내 고립된 fat-HU 픽셀 → TISSUE_ARTIFACT로 명시적 표기.

    Classes: BG(0), FAT(1), SOFT(2), BONE(3), FLUID(4), VESSEL(5), ARTIFACT(6)
    """
    H, W = image_hu.shape
    label_map = np.full((H, W), TISSUE_BG, dtype=np.int32)

    # === Feature 추출 ===
    local_mean = features['local_mean']
    std_7 = features['std_7']
    dist_bg = features['dist_from_bg']
    anisotropy = features.get('anisotropy', np.zeros((H, W), dtype=np.float32))
    kurtosis = features.get('kurtosis', np.zeros((H, W), dtype=np.float32))
    local_range = features.get('local_range', np.full((H, W), 100, dtype=np.float32))
    vesselness = features.get('vesselness', np.zeros((H, W), dtype=np.float32))
    connectivity = features.get('connectivity', np.zeros((H, W), dtype=np.float32))

    # === Spatial context: artifact가 발생하는 물리적 영역 ===
    arm_band = features.get('arm_band', np.zeros((H, W), dtype=bool))
    starvation = features.get('starvation_mask', np.zeros((H, W), dtype=bool))
    between_bones = features.get('between_bones', np.zeros((H, W), dtype=np.float32))
    artifact_zone = arm_band | starvation | (between_bones >= 1)

    in_body = body_mask & (dist_bg > 2)

    # === 1단계: Bone (HU > 200, 명확한 threshold) ===
    bone = in_body & (image_hu > 200)
    label_map[bone] = TISSUE_BONE

    # 나머지 body 픽셀
    remaining = in_body & ~bone

    # === 1.5단계: Artifact — 심하게 손상된 영역 명시적 표기 ===
    # 조건1: artifact_zone 내에서 HU가 심하게 떨어진 영역 (photon starvation)
    severe_artifact = remaining & artifact_zone & (image_hu < -200)
    label_map[severe_artifact] = TISSUE_ARTIFACT

    # 조건2: artifact_zone 내 fat-HU 범위이지만 connectivity 낮은 고립 픽셀
    # → 진짜 fat이 아니라 artifact로 HU가 떨어진 조직
    fat_hu_in_zone = remaining & artifact_zone & (local_mean > -180) & (local_mean < -30)
    isolated_in_zone = fat_hu_in_zone & (connectivity < 0.3)
    label_map[isolated_in_zone] = TISSUE_ARTIFACT

    # 조건3: artifact_zone 내 fat-HU + 높은 anisotropy/kurtosis (streak 특성)
    streak_in_zone = fat_hu_in_zone & (connectivity >= 0.3) & (
        (anisotropy > 0.3) | (kurtosis > 3.0) | (local_range > 120)
    )
    label_map[streak_in_zone] = TISSUE_ARTIFACT

    # artifact로 분류된 픽셀 제거
    remaining = remaining & (label_map == TISSUE_BG)

    # === 2단계: 각 class별 confidence score 계산 ===

    # --- Fat confidence ---
    # artifact_zone 밖에서만 fat 후보 (zone 안은 이미 ARTIFACT로 처리)
    # + connectivity가 높은(큰 blob) 것만 fat
    is_fat_hu = remaining & (local_mean > -180) & (local_mean < -30) & ~artifact_zone
    fat_conf = np.zeros((H, W), dtype=np.float32)
    m = is_fat_hu
    if m.any():
        f1 = 1.0 - np.clip(anisotropy[m] / 0.4, 0, 1)         # 등방 → 1
        f2 = 1.0 - np.clip(std_7[m] / 40.0, 0, 1)              # 균일 → 1
        f3 = 1.0 - np.clip(kurtosis[m] / 6.0, 0, 1)            # Gaussian → 1
        f4 = 1.0 - np.clip(local_range[m] / 100.0, 0, 1)       # 진폭 작음 → 1
        f5 = np.clip(connectivity[m], 0, 1)                      # 큰 blob → 1
        fat_conf[m] = f1 * 0.25 + f2 * 0.20 + f3 * 0.15 + f4 * 0.20 + f5 * 0.20

    # --- Fluid confidence ---
    is_fluid_hu = remaining & (local_mean >= -10) & (local_mean < 30)
    fluid_conf = np.zeros((H, W), dtype=np.float32)
    m = is_fluid_hu
    if m.any():
        f1 = 1.0 - np.clip(std_7[m] / 25.0, 0, 1)
        f2 = 1.0 - np.clip(anisotropy[m] / 0.3, 0, 1)
        f3 = 1.0 - np.clip(local_range[m] / 50.0, 0, 1)
        f4 = 1.0 - np.clip(vesselness[m] / 0.3, 0, 1)
        fluid_conf[m] = f1 * 0.30 + f2 * 0.20 + f3 * 0.30 + f4 * 0.20

    # --- Vessel confidence ---
    is_vessel_hu = remaining & (local_mean >= 20) & (local_mean < 80)
    vessel_conf = np.zeros((H, W), dtype=np.float32)
    m = is_vessel_hu
    if m.any():
        f1 = np.clip(vesselness[m] / 0.3, 0, 1)
        f2 = 1.0 - np.clip(anisotropy[m] / 0.3, 0, 1)
        f3 = 1.0 - np.clip(std_7[m] / 40.0, 0, 1)
        f4 = 1.0 - np.clip(kurtosis[m] / 4.0, 0, 1)
        vessel_conf[m] = f1 * 0.45 + f2 * 0.25 + f3 * 0.15 + f4 * 0.15

    # === 3단계: Confidence 기반 할당 ===
    MIN_CONF = 0.4

    # Fat: fat HU 범위 + confidence 충분
    label_map[is_fat_hu & (fat_conf > MIN_CONF)] = TISSUE_FAT
    # Fat HU 범위인데 confidence 부족 → artifact-corrupted soft tissue
    label_map[is_fat_hu & (fat_conf <= MIN_CONF)] = TISSUE_SOFT

    # Fluid
    is_fluid = is_fluid_hu & (fluid_conf > MIN_CONF) & (label_map == TISSUE_BG)
    label_map[is_fluid] = TISSUE_FLUID

    # Vessel
    VESSEL_MIN_CONF = 0.5
    is_vessel = is_vessel_hu & (vessel_conf > VESSEL_MIN_CONF) & (label_map == TISSUE_BG)
    label_map[is_vessel] = TISSUE_VESSEL

    # === 4단계: 나머지 body 픽셀 → soft tissue ===
    still_bg = in_body & (label_map == TISSUE_BG)
    label_map[still_bg] = TISSUE_SOFT

    # === 5단계: Small component cleanup ===
    min_sizes = {
        TISSUE_FAT: 30, TISSUE_SOFT: 30, TISSUE_BONE: 30,
        TISSUE_FLUID: 80, TISSUE_VESSEL: 50, TISSUE_ARTIFACT: 20,
    }
    for tid, min_sz in min_sizes.items():
        mask = label_map == tid
        labeled_m, n_feat = label(mask)
        if n_feat == 0:
            continue
        comp_sizes = np.bincount(labeled_m.ravel())
        comp_sizes[0] = 0
        for i in range(1, len(comp_sizes)):
            if comp_sizes[i] < min_sz:
                label_map[labeled_m == i] = TISSUE_BG

    # === 6단계: Cleanup 후 미분류 → soft tissue ===
    final_bg = in_body & (label_map == TISSUE_BG)
    label_map[final_bg] = TISSUE_SOFT

    return label_map


# ============================================================
# 2. Artifact mask
# ============================================================

def compute_artifact_mask(image_hu, features, body_mask, label_map):
    """
    Streak artifact probability map (0-1)

    v3 개선:
    1. 폐/공기 억제: HU < -200 영역은 artifact가 아님 (늑골 근처 폐 오분류 방지)
    2. Tissue-aware: 연조직/지방 범위(-200~200)만 artifact 후보
    3. Arm band 정밀화: base 확률 제거, gradient/deviation 증거 필수
    4. Between-bones 골반 범위 확대
    """
    H, W = image_hu.shape

    dist_bone = features['dist_from_bone']
    grad_mag = features['grad_mag']
    dir_cons = features['dir_consistency']
    hu_dev = features['hu_deviation']
    between_bones = features['between_bones']

    not_bone = label_map != TISSUE_BONE

    # === v3: Tissue-aware suppression ===
    # 폐/공기(HU < -200)는 streak artifact가 아님 → 강하게 억제
    # 연조직/지방 범위(-200~200)에서만 artifact 탐지
    tissue_factor = np.clip((image_hu + 200) / 100, 0, 1)  # -200 이하 → 0, -100 이상 → 1
    # 뼈(>200)도 이미 not_bone으로 제외되므로 여기선 상관없음

    # === Source 1: Proximity-based artifact ===
    proximity_factor = 1.0 / (1.0 + np.exp((dist_bone - 30) / 8.0))
    gradient_factor = 1.0 / (1.0 + np.exp(-(grad_mag - 100) / 60.0))
    deviation_factor = 1.0 / (1.0 + np.exp(-(hu_dev - 25) / 15.0))
    direction_factor = np.clip((dir_cons - 0.15) / 0.25, 0, 1)

    proximity_artifact = (
        proximity_factor *
        np.maximum(gradient_factor, deviation_factor) *
        (0.5 + 0.5 * direction_factor)
    ).astype(np.float32)

    # === Source 2: Between-bones artifact ===
    # v3: between_bones range는 이미 80으로 확대됨
    between_artifact = np.clip(between_bones.astype(np.float32) / 1.5, 0, 1)
    between_artifact *= np.clip(1.0 - (dist_bone - 60) / 40, 0, 1)

    # === Source 3: Arm artifact band (v3 정밀화) ===
    # v3: 팔뼈 거리 감쇠 + 증거 기반
    # arm_dist_weight: 팔뼈에 가까울수록 1, 척추 쪽으로 갈수록 0
    arm_band = features.get('arm_band', np.zeros(image_hu.shape, dtype=bool))
    arm_dw = features.get('arm_dist_weight', np.zeros(image_hu.shape, dtype=np.float32))
    # 증거 기반 × 거리 가중치: 팔뼈 가까이 + 증거 있으면 artifact
    arm_band_enhanced = (
        arm_dw * gradient_factor * 0.6 +
        arm_dw * deviation_factor * 0.4
    )
    # 팔뼈 바로 근처는 최소 확률 부여 (거리 가중치가 높은 곳만)
    arm_band_min = arm_dw * 0.15
    arm_band_enhanced = np.maximum(arm_band_enhanced, arm_band_min)

    # === Source 4: Photon starvation (v3) ===
    # Arm 사이 X선 감쇠로 인한 어두운 밴드 → 강한 artifact
    starvation_weight = features.get('starvation_weight', np.zeros(image_hu.shape, dtype=np.float32))
    starvation_mask = features.get('starvation_mask', np.zeros(image_hu.shape, dtype=bool))

    # === 결합: 넷 중 가장 높은 값 ===
    artifact_prob = np.maximum(proximity_artifact, between_artifact)
    artifact_prob = np.maximum(artifact_prob, arm_band_enhanced)
    artifact_prob = np.maximum(artifact_prob, starvation_weight * 0.8)

    # === Multi-feature artifact boost ===
    # 축3: Anisotropy — 방향성 높으면 boost
    anisotropy = features.get('anisotropy', np.zeros(image_hu.shape, dtype=np.float32))
    aniso_boost = 0.8 + 0.4 * anisotropy

    # 축4: Kurtosis — heavy-tail이면 boost (streak의 밝/어 교대 패턴)
    kurtosis = features.get('kurtosis', np.zeros(image_hu.shape, dtype=np.float32))
    kurt_boost = 0.9 + 0.2 * np.clip(kurtosis / 5.0, 0, 1)

    artifact_prob *= aniso_boost * kurt_boost

    # === v3: Tissue-aware + Body + Not-bone 적용 ===
    artifact_prob *= body_mask.astype(np.float32)
    artifact_prob *= not_bone.astype(np.float32)
    # Tissue factor: 폐/공기 억제, BUT starvation 영역은 면제
    # (starvation은 HU가 낮아 공기처럼 보이지만 실제로는 감쇠된 조직)
    tissue_factor_adjusted = np.where(starvation_mask, 1.0, tissue_factor)
    artifact_prob *= tissue_factor_adjusted  # 폐/공기 영역 억제 (starvation 제외)

    # === v3: 병변 보호 ===
    # Streak artifact는 줄무늬 패턴(높은 std, 비균일)
    # 병변/종양은 균일한 덩어리(낮은 std_15, 큰 connected component)
    # → 균일하고 큰 영역은 artifact에서 제외
    std_15 = features['std_15']
    # 균일도 기반 보호: std_15 < 30이면 균일한 덩어리 (병변/장기) → artifact 억제
    uniformity_protection = np.clip((std_15 - 15) / 30, 0, 1)  # std_15=15이하→0(보호), 45이상→1(그대로)
    # 뼈 바로 근처(dist<10)는 보호하지 않음 (진짜 streak일 가능성 높음)
    near_bone = np.clip(1.0 - dist_bone / 10, 0, 1)
    protection = uniformity_protection + near_bone  # 뼈 근처이거나 비균일하면 보호 안 함
    protection = np.clip(protection, 0, 1)
    artifact_prob *= protection

    # 약한 smoothing
    artifact_prob = uniform_filter(artifact_prob.astype(np.float64), size=3).astype(np.float32)

    return artifact_prob


# ============================================================
# 3. Fluid mask
# ============================================================

def compute_fluid_mask(image_hu, features, body_mask):
    """
    Fluid (유체) 영역 검출 - between-bones artifact 제외

    핵심 규칙 (데이터 분석 결과):
    - FLUID: between_bones == 0 AND std_15 < 45 AND component ≥ 80px
    - ARTIFACT: between_bones ≥ 1 (뼈 쌍 사이 = streak artifact)
    - NOISE: 작은 component (<30px) 또는 높은 std

    데이터 근거:
    - Real fluid comp 953: 2345px, std15=21.8, dist_bone=44.7, between=0
    - Artifact comp 861: 143px, std15=143.9, dist_bone=5.8, between=1
    - Artifact comp 104: 65px, std15=24.4, dist_bone=9.5, between=1
    """
    H, W = image_hu.shape
    between_bones = features['between_bones']

    # Step 1: Fluid 후보 (수분 HU 범위 + body 내부)
    candidate = (
        body_mask &
        (image_hu >= -30) & (image_hu < 30) &
        (features['std_15'] < 60)
    )

    # Step 2a: Between-bones 영역 제외 (뼈쌍 사이 streak artifact)
    candidate = candidate & (between_bones < 1)

    # Step 2b: Arm artifact band 영역 제외 (팔뼈 사이 수평 artifact)
    arm_band = features.get('arm_band', np.zeros_like(image_hu, dtype=bool))
    candidate = candidate & ~arm_band

    # Step 2c: 높은 anisotropy 영역 제외 (방향성 streak → fluid 아님)
    anisotropy = features.get('anisotropy', np.zeros_like(image_hu, dtype=np.float32))
    candidate = candidate & (anisotropy < 0.5)

    # Step 2d: 관형 구조(vessel) 제외 — fluid는 pool 형태, vessel은 tube 형태
    vesselness = features.get('vesselness', np.zeros_like(image_hu, dtype=np.float32))
    candidate = candidate & (vesselness < 0.3)

    # Step 3: Connected component → 큰 영역만 유지
    labeled, n_comp = label(candidate)
    fluid_mask = np.zeros((H, W), dtype=np.float32)

    if n_comp > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0

        for i in range(1, len(sizes)):
            comp_mask = labeled == i

            if sizes[i] >= 80:
                # 큰 영역 + NOT between bones → 높은 확률 fluid
                mean_std15 = np.mean(features['std_15'][comp_mask])
                mean_dist_bone = np.mean(features['dist_from_bone'][comp_mask])

                # Confidence: 균일할수록, bone에서 멀수록 높음
                conf_std = np.clip(1.0 - (mean_std15 - 15) / 60, 0.3, 1.0)
                conf_dist = np.clip(mean_dist_bone / 50, 0.3, 1.0)
                confidence = conf_std * 0.7 + conf_dist * 0.3

                fluid_mask[comp_mask] = confidence

            elif sizes[i] >= 30:
                # 중간 영역: bone에서 멀고 균일하면 fluid
                mean_std15 = np.mean(features['std_15'][comp_mask])
                mean_dist_bone = np.mean(features['dist_from_bone'][comp_mask])

                if mean_std15 < 40 and mean_dist_bone > 20:
                    confidence = np.clip(1.0 - (mean_std15 - 15) / 50, 0.2, 0.7)
                    fluid_mask[comp_mask] = confidence

    return fluid_mask


# ============================================================
# 4. Noise level map
# ============================================================

def compute_noise_level_map(image_hu, features, body_mask, artifact_prob):
    """
    각 픽셀의 추정 노이즈 레벨 (denoise 강도 가이드)

    원리:
    - std_3: fine-scale variability (noise + edge)
    - std_15: large-scale variability (structure)
    - noise_component ≈ std_3 - contribution_from_structure

    데이터:
    - Soft tissue (uniform): std_3=18.5 → baseline noise
    - Fat (uniform): std_3=19.8 → similar noise
    - Near bone: std_3=52.7 → artifact + noise
    - Boundary: std_3=35+ → edge contribution

    방법: std_3에서 edge 기여분을 빼서 순수 noise 추정
    """
    std_3 = features['std_3']
    std_15 = features['std_15']
    grad_mag = features['grad_mag']

    # Edge contribution 추정: gradient가 높으면 std_3가 높아지지만 이는 noise가 아님
    # Simple model: noise = std_3 - alpha * normalized_gradient
    grad_norm = np.clip(grad_mag / 500, 0, 1)  # normalize

    # Noise level = std_3에서 structure/edge 기여분 제거
    # Structure 기여분 ≈ min(std_3, std_15 * 0.5)
    structure_contribution = np.minimum(std_3, std_15 * 0.4)
    noise_estimate = np.maximum(std_3 - structure_contribution * grad_norm, 0)

    # 정규화: body 영역의 중앙값으로 정규화
    body_noise = noise_estimate[body_mask]
    if len(body_noise) > 0:
        noise_median = np.median(body_noise)
        if noise_median > 0:
            noise_normalized = noise_estimate / noise_median
        else:
            noise_normalized = noise_estimate
    else:
        noise_normalized = noise_estimate

    # Artifact 영역은 noise level 높임 (aggressive denoising 필요)
    noise_normalized += artifact_prob * 0.5

    # Body 외부는 0
    noise_normalized *= body_mask.astype(np.float32)

    # Clip to reasonable range
    noise_normalized = np.clip(noise_normalized, 0, 3.0)

    return noise_normalized.astype(np.float32)


# ============================================================
# 5. Structure preservation mask
# ============================================================

def compute_structure_mask(image_hu, features, body_mask, label_map,
                           artifact_prob, fluid_mask, noise_level):
    """
    보존해야 할 구조 마스크 (0-1)

    높은 값 = 반드시 보존 (edge, organ boundary, vessel)
    낮은 값 = denoise 가능 (균일한 내부)

    구성:
    - Base: body 내부 = 0.5
    - Edge boost: gradient 높으면 보존도 높임
    - Artifact penalty: artifact 영역은 보존 안 함
    - Noise penalty: 노이즈 높으면 보존보다 denoise 우선
    """
    grad_mag = features['grad_mag']

    # Base: body 내부
    structure = body_mask.astype(np.float32) * 0.5

    # Edge/boundary boost: gradient 높으면 보존 강화
    # 데이터: soft tissue grad_mag median=130, bone=1443
    edge_factor = np.clip(grad_mag / 300, 0, 0.5)
    structure += edge_factor

    # Fluid 영역: 반드시 보존 (잘못 지우면 안 됨)
    structure = np.maximum(structure, fluid_mask * 0.8)

    # Vessel 영역: 해부학적 구조 보존
    structure[label_map == TISSUE_VESSEL] = np.maximum(
        structure[label_map == TISSUE_VESSEL], 0.85
    )

    # Artifact penalty: artifact일 확률이 높으면 보존하지 않음
    structure *= (1.0 - artifact_prob * 0.8)

    # Bone: 구조 자체는 보존 (artifact는 bone 주변에서 발생)
    structure[label_map == TISSUE_BONE] = 0.9

    # Body 외부는 0
    structure *= body_mask.astype(np.float32)

    return np.clip(structure, 0, 1).astype(np.float32)


# ============================================================
# Main pipeline
# ============================================================

def generate_action_masks(image_hu, verbose=True):
    """
    전체 파이프라인: Action-Guided Mask 생성

    Args:
        image_hu: (H, W) float32 HU 값
        verbose: True면 상세 출력, False면 무출력

    Returns:
        masks: dict with keys:
            'tissue_label': int32 (H,W) - 0:bg, 1:fat, 2:soft, 3:bone, 4:fluid, 5:vessel, 6:artifact
            'structure': float32 (H,W) 0-1 - 보존 강도
            'artifact': float32 (H,W) 0-1 - artifact 확률
            'fluid': float32 (H,W) 0-1 - fluid 확률
            'noise_level': float32 (H,W) 0-3 - 추정 노이즈 레벨
        features: dict of computed features (std, anisotropy, kurtosis, local_range, vesselness 등)
    """
    _p = print if verbose else (lambda *a, **k: None)

    _p("=" * 55)
    _p("Action-Guided Mask Generation")
    _p("=" * 55)

    H, W = image_hu.shape
    _p(f"  Image: {H}x{W}, HU: [{image_hu.min():.0f}, {image_hu.max():.0f}]")

    # Feature extraction
    _p("  Step 1: Computing features...")
    features = compute_action_features(image_hu)

    # Body detection
    _p("  Step 2: Body detection...")
    body_mask = find_body_mask(image_hu)
    _p(f"    Body: {body_mask.sum()} px ({body_mask.mean()*100:.1f}%)")

    # Arm artifact band detection (v3: distance weight 포함)
    _p("  Step 2b: Arm artifact band detection...")
    arm_band, arm_dist_weight = detect_arm_artifact_band(image_hu, body_mask, margin=30)
    features['arm_band'] = arm_band
    features['arm_dist_weight'] = arm_dist_weight
    _p(f"    Arm band: {arm_band.sum()} px")

    # Photon starvation detection (v3: 팔 사이 X선 감쇠 밴드)
    _p("  Step 2c: Photon starvation detection...")
    starvation_mask, starvation_weight = detect_photon_starvation(image_hu, body_mask, arm_band)
    features['starvation_mask'] = starvation_mask
    features['starvation_weight'] = starvation_weight
    _p(f"    Starvation: {starvation_mask.sum()} px")

    # Local connectivity (fat blob vs artifact)
    _p("  Step 2d: Local connectivity...")
    connectivity = compute_local_connectivity(image_hu, body_mask)
    features['connectivity'] = connectivity
    _p(f"    Fat-HU connected blobs: {(connectivity > 0.5).sum()} px")

    # Tissue classification
    _p("  Step 3: Tissue classification...")
    tissue_label = classify_tissue(image_hu, features, body_mask)
    for tid, tname in TISSUE_NAMES.items():
        count = (tissue_label == tid).sum()
        _p(f"    {tname}: {count} px")

    # Artifact mask
    _p("  Step 4: Artifact detection...")
    artifact_prob = compute_artifact_mask(image_hu, features, body_mask, tissue_label)
    artifact_strong = (artifact_prob > 0.3).sum()
    _p(f"    Artifact pixels (prob > 0.3): {artifact_strong}")

    # Fluid mask
    _p("  Step 5: Fluid detection...")
    fluid_mask = compute_fluid_mask(image_hu, features, body_mask)
    fluid_count = (fluid_mask > 0.3).sum()
    _p(f"    Fluid pixels (prob > 0.3): {fluid_count}")

    # Noise level
    _p("  Step 6: Noise estimation...")
    noise_level = compute_noise_level_map(image_hu, features, body_mask, artifact_prob)
    body_noise = noise_level[body_mask]
    _p(f"    Body noise level: median={np.median(body_noise):.2f}, "
       f"mean={np.mean(body_noise):.2f}")

    # Structure preservation
    _p("  Step 7: Structure preservation mask...")
    structure = compute_structure_mask(image_hu, features, body_mask,
                                       tissue_label, artifact_prob, fluid_mask, noise_level)

    masks = {
        'tissue_label': tissue_label,
        'structure': structure,
        'artifact': artifact_prob,
        'fluid': fluid_mask,
        'noise_level': noise_level,
        'body': body_mask.astype(np.float32),
        'arm_band': features.get('arm_band', np.zeros(image_hu.shape, dtype=bool)),
    }

    _p("=" * 55)
    return masks, features


# ============================================================
# Visualization
# ============================================================

def visualize_action_masks(image_hu, masks, output_dir):
    """종합 시각화"""
    os.makedirs(output_dir, exist_ok=True)

    def orient(arr):
        return np.flipud(np.rot90(arr, k=1))

    # CT display
    hu_disp = np.clip((image_hu + 160) / 400, 0, 1)
    ct_uint8 = (hu_disp * 255).astype(np.uint8)

    # ============================================================
    # Figure 1: 5-panel overview
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # CT
    axes[0, 0].imshow(orient(ct_uint8), cmap='gray')
    axes[0, 0].set_title('Original CT', fontsize=13)
    axes[0, 0].axis('off')

    # Tissue classification
    tissue_colors = np.zeros((*image_hu.shape, 3), dtype=np.uint8)
    tissue_colors[masks['tissue_label'] == 1] = [255, 200, 50]   # fat: yellow
    tissue_colors[masks['tissue_label'] == 2] = [255, 100, 100]  # soft: red
    tissue_colors[masks['tissue_label'] == 3] = [150, 150, 255]  # bone: blue
    tissue_colors[masks['tissue_label'] == 4] = [50, 200, 255]   # fluid: cyan
    tissue_colors[masks['tissue_label'] == 5] = [255, 50, 200]   # vessel: magenta
    tissue_colors[masks['tissue_label'] == 6] = [200, 50, 50]    # artifact: dark red
    axes[0, 1].imshow(orient(tissue_colors))
    axes[0, 1].set_title('Tissue Classification\n(Yel=Fat, Red=Soft, Blu=Bone, DkRed=Artifact)', fontsize=11)
    axes[0, 1].axis('off')

    # Structure preservation
    im = axes[0, 2].imshow(orient(masks['structure']), cmap='RdYlGn', vmin=0, vmax=1)
    axes[0, 2].set_title('Structure Preservation\n(Green=preserve, Red=removable)', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    # Artifact mask
    im = axes[1, 0].imshow(orient(masks['artifact']), cmap='Reds', vmin=0, vmax=0.5)
    axes[1, 0].set_title(f'Artifact Probability\n({(masks["artifact"]>0.3).sum()} px > 0.3)', fontsize=12)
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    # Fluid mask
    im = axes[1, 1].imshow(orient(masks['fluid']), cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Fluid Regions\n({(masks["fluid"]>0.3).sum()} px > 0.3)', fontsize=12)
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    # Noise level
    im = axes[1, 2].imshow(orient(masks['noise_level']), cmap='hot', vmin=0, vmax=2)
    axes[1, 2].set_title('Noise Level Map\n(Denoise intensity guide)', fontsize=12)
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.suptitle('Action-Guided Mask System', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_masks_overview.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================================
    # Figure 2: CT + overlay comparison
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    ct_rgb = np.stack([ct_uint8]*3, axis=-1)

    # Original
    axes[0].imshow(orient(ct_rgb))
    axes[0].set_title('Original CT', fontsize=13)
    axes[0].axis('off')

    # CT + Artifact overlay (red)
    overlay1 = ct_rgb.copy().astype(np.float32)
    artifact_color = np.zeros_like(ct_rgb, dtype=np.float32)
    artifact_color[:, :, 0] = 255  # Red
    alpha1 = masks['artifact'][:, :, np.newaxis] * 0.6
    overlay1 = overlay1 * (1 - alpha1) + artifact_color * alpha1
    axes[1].imshow(orient(overlay1.astype(np.uint8)))
    axes[1].set_title('CT + Artifact (Red overlay)', fontsize=13)
    axes[1].axis('off')

    # CT + Fluid overlay (blue) + Structure edges (green)
    overlay2 = ct_rgb.copy().astype(np.float32)
    # Fluid: blue
    fluid_color = np.zeros_like(ct_rgb, dtype=np.float32)
    fluid_color[:, :, 2] = 255
    alpha2 = masks['fluid'][:, :, np.newaxis] * 0.5
    overlay2 = overlay2 * (1 - alpha2) + fluid_color * alpha2
    axes[2].imshow(orient(overlay2.astype(np.uint8)))
    axes[2].set_title('CT + Fluid (Blue overlay)', fontsize=13)
    axes[2].axis('off')

    plt.suptitle('Action Mask Overlays on CT', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_masks_overlays.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ============================================================
    # Figure 3: Combined action map
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Action map
    H, W = image_hu.shape
    action_map = np.zeros((H, W, 3), dtype=np.uint8)

    # Background: black
    # Structure (preserve): green intensity = structure value
    structure_val = masks['structure']
    action_map[:, :, 1] = (structure_val * 200).astype(np.uint8)

    # Artifact: red (overrides green)
    artifact_val = masks['artifact']
    artifact_strong_mask = artifact_val > 0.15
    action_map[artifact_strong_mask, 0] = (artifact_val[artifact_strong_mask] * 255).astype(np.uint8)
    action_map[artifact_strong_mask, 1] = 0  # remove green

    # Fluid: blue (add to existing)
    fluid_val = masks['fluid']
    fluid_mask_bool = fluid_val > 0.2
    action_map[fluid_mask_bool, 2] = (fluid_val[fluid_mask_bool] * 255).astype(np.uint8)

    axes[0].imshow(orient(action_map))
    axes[0].set_title('Combined Action Map\nGreen=preserve, Red=artifact, Blue=fluid', fontsize=12)
    axes[0].axis('off')

    # Denoise intensity guide
    # Combine: high noise + artifact = aggressive denoise
    # Low noise + structure = gentle/no denoise
    denoise_guide = masks['noise_level'].copy()
    denoise_guide += masks['artifact'] * 1.0  # artifact 영역 더 강한 denoise
    denoise_guide *= (1.0 - masks['structure'] * 0.5)  # structure 영역은 약한 denoise
    denoise_guide *= masks['body']  # body 내부만
    denoise_guide = np.clip(denoise_guide, 0, 3)

    im = axes[1].imshow(orient(denoise_guide), cmap='YlOrRd', vmin=0, vmax=2.5)
    axes[1].set_title('Denoise Intensity Guide\n(Yellow=gentle, Red=aggressive)', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.suptitle('Model Guidance Maps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_masks_guide.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save individual masks as images
    for name in ['structure', 'artifact', 'fluid', 'noise_level', 'body']:
        mask = masks[name]
        if name == 'noise_level':
            mask_norm = np.clip(mask / 3.0, 0, 1)
        else:
            mask_norm = np.clip(mask, 0, 1)
        mask_uint8 = orient((mask_norm * 255).astype(np.uint8))
        Image.fromarray(mask_uint8).save(f"{output_dir}/mask_{name}.png")

    # Save tissue label
    tissue_uint8 = orient((masks['tissue_label'].astype(np.float32) / 3 * 255).astype(np.uint8))
    Image.fromarray(tissue_uint8).save(f"{output_dir}/mask_tissue_label.png")

    print(f"  All visualizations saved to: {output_dir}")


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    import nibabel as nib

    patient_id = "1728852"
    slice_idx = 27
    nifti_path = f"F:/LD-CT SR/Data/NC-CT NIfTI/{patient_id}.nii"
    output_dir = "F:/LD-CT SR/Outputs/ma_hybrid/action_masks"

    print(f"Loading: {nifti_path}")
    nifti_img = nib.load(nifti_path)
    volume = nifti_img.get_fdata()
    image_hu = volume[:, :, slice_idx].astype(np.float32)

    masks, features = generate_action_masks(image_hu)
    visualize_action_masks(image_hu, masks, output_dir)
