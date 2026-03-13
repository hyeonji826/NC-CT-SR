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

TISSUE_NAMES = {0: 'background', 1: 'fat', 2: 'soft_tissue', 3: 'bone'}
TISSUE_COLORS = {
    0: [0, 0, 0],        # background: black
    1: [255, 200, 50],    # fat: yellow
    2: [255, 100, 100],   # soft tissue: red
    3: [150, 150, 255],   # bone: blue
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
# 1. Tissue classification (simplified)
# ============================================================

def classify_tissue(image_hu, features, body_mask):
    """
    기본 조직 분류 (4 class)
    데이터 기반 threshold 사용
    """
    H, W = image_hu.shape
    label_map = np.full((H, W), TISSUE_BG, dtype=np.int32)

    local_mean = features['local_mean']
    local_std = features['std_7']
    dist_bg = features['dist_from_bg']

    # Body 내부만 분류
    in_body = body_mask & (dist_bg > 2)

    # Bone: HU > 200 (넉넉하게)
    bone = in_body & (image_hu > 200)
    label_map[bone] = TISSUE_BONE

    # Fat: HU [-180, -30], body 내부
    fat = in_body & (local_mean > -180) & (local_mean < -30) & ~bone
    label_map[fat] = TISSUE_FAT

    # Soft tissue: HU [-30, 200], body 내부
    soft = in_body & (local_mean >= -30) & (local_mean <= 200) & ~bone
    label_map[soft] = TISSUE_SOFT

    # 미분류 body 픽셀 → HU 기반 할당
    unassigned = in_body & (label_map == TISSUE_BG)
    label_map[unassigned & (image_hu < -30)] = TISSUE_FAT
    label_map[unassigned & (image_hu >= -30)] = TISSUE_SOFT

    # Connected component cleanup: 작은 region 제거 후 주변으로 할당
    for tid in [TISSUE_FAT, TISSUE_SOFT, TISSUE_BONE]:
        mask = label_map == tid
        labeled_m, n_feat = label(mask)
        if n_feat == 0:
            continue
        comp_sizes = np.bincount(labeled_m.ravel())
        comp_sizes[0] = 0
        for i in range(1, len(comp_sizes)):
            if comp_sizes[i] < 30:
                label_map[labeled_m == i] = TISSUE_BG  # 일단 미분류

    # 미분류 body 픽셀 다시 할당 (nearest neighbor)
    unassigned = in_body & (label_map == TISSUE_BG)
    if unassigned.any():
        # 각 class까지의 거리
        for tid in [TISSUE_FAT, TISSUE_SOFT, TISSUE_BONE]:
            assigned = label_map == tid
            if not assigned.any():
                continue
            dist = ndimage.distance_transform_edt(~assigned)
            # unassigned 중 이 class가 가장 가까운 픽셀
            # HU 유사도 가중
            hu_center = {TISSUE_FAT: -90, TISSUE_SOFT: 40, TISSUE_BONE: 500}[tid]
            score = dist + np.abs(image_hu - hu_center) * 0.2
            # 이미 할당된 곳은 제외
            score[~unassigned] = 1e9
            # 가장 가까운 class로 할당 (greedy)

        # 더 간단한 방법: HU 기반으로 할당
        still_unassigned = in_body & (label_map == TISSUE_BG)
        label_map[still_unassigned & (image_hu < -30)] = TISSUE_FAT
        label_map[still_unassigned & (image_hu >= -30) & (image_hu < 200)] = TISSUE_SOFT
        label_map[still_unassigned & (image_hu >= 200)] = TISSUE_BONE

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

def generate_action_masks(image_hu):
    """
    전체 파이프라인: Action-Guided Mask 생성

    Returns:
        masks: dict with keys:
            'tissue_label': int32 (H,W) - 0:bg, 1:fat, 2:soft, 3:bone
            'structure': float32 (H,W) 0-1 - 보존 강도
            'artifact': float32 (H,W) 0-1 - artifact 확률
            'fluid': float32 (H,W) 0-1 - fluid 확률
            'noise_level': float32 (H,W) 0-3 - 추정 노이즈 레벨
        features: dict of computed features
    """
    print("=" * 55)
    print("Action-Guided Mask Generation")
    print("=" * 55)

    H, W = image_hu.shape
    print(f"  Image: {H}x{W}, HU: [{image_hu.min():.0f}, {image_hu.max():.0f}]")

    # Feature extraction
    print("  Step 1: Computing features...")
    features = compute_action_features(image_hu)

    # Body detection
    print("  Step 2: Body detection...")
    body_mask = find_body_mask(image_hu)
    print(f"    Body: {body_mask.sum()} px ({body_mask.mean()*100:.1f}%)")

    # Arm artifact band detection (v3: distance weight 포함)
    print("  Step 2b: Arm artifact band detection...")
    arm_band, arm_dist_weight = detect_arm_artifact_band(image_hu, body_mask, margin=30)
    features['arm_band'] = arm_band
    features['arm_dist_weight'] = arm_dist_weight
    print(f"    Arm band: {arm_band.sum()} px")

    # Photon starvation detection (v3: 팔 사이 X선 감쇠 밴드)
    print("  Step 2c: Photon starvation detection...")
    starvation_mask, starvation_weight = detect_photon_starvation(image_hu, body_mask, arm_band)
    features['starvation_mask'] = starvation_mask
    features['starvation_weight'] = starvation_weight
    print(f"    Starvation: {starvation_mask.sum()} px")

    # Tissue classification
    print("  Step 3: Tissue classification...")
    tissue_label = classify_tissue(image_hu, features, body_mask)
    for tid, tname in TISSUE_NAMES.items():
        count = (tissue_label == tid).sum()
        print(f"    {tname}: {count} px")

    # Artifact mask
    print("  Step 4: Artifact detection...")
    artifact_prob = compute_artifact_mask(image_hu, features, body_mask, tissue_label)
    artifact_strong = (artifact_prob > 0.3).sum()
    print(f"    Artifact pixels (prob > 0.3): {artifact_strong}")

    # Fluid mask
    print("  Step 5: Fluid detection...")
    fluid_mask = compute_fluid_mask(image_hu, features, body_mask)
    fluid_count = (fluid_mask > 0.3).sum()
    print(f"    Fluid pixels (prob > 0.3): {fluid_count}")

    # Noise level
    print("  Step 6: Noise estimation...")
    noise_level = compute_noise_level_map(image_hu, features, body_mask, artifact_prob)
    body_noise = noise_level[body_mask]
    print(f"    Body noise level: median={np.median(body_noise):.2f}, "
          f"mean={np.mean(body_noise):.2f}")

    # Structure preservation
    print("  Step 7: Structure preservation mask...")
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

    print("=" * 55)
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
    axes[0, 1].imshow(orient(tissue_colors))
    axes[0, 1].set_title('Tissue Classification\n(Yellow=Fat, Red=Soft, Blue=Bone)', fontsize=12)
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
