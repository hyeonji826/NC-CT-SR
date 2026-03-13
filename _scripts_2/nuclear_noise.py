# -*- coding: utf-8 -*-
"""
nuclear_noise.py
================
핵의학 Abd NC-CT NPS 기반 synthetic noise 모델

변경 이력:
- v1: 초기 구현 (_scripts_7)
- v2: DC clipping 추가, streak artifact 제거 (1/r 발산 문제)
- v3: _scripts 통합 (hu_min/hu_max 기본값 -1000/1000)
- v4: 아티팩트 타겟 노이즈 추가 (artifact 영역에 강한 노이즈)
"""
import os
import numpy as np
import yaml


class NuclearNoiseModel:
    """
    핵의학 Abd NC-CT NPS 기반 synthetic noise 모델
    - NPS radial + freqs로 frequency shaping
    - target_noise_std_hu 로 std 맞춤
    - 선택적으로 beam-hardening(cupping) 추가
    - DC 성분 clipping으로 저주파 과잉 방지
    """

    def __init__(self,
                 nps_root: str,
                 target_noise_std_hu: float = 29.4,
                 hu_min: float = -1000.0,
                 hu_max: float = 1000.0,
                 beam_hardening_strength: float = 0.15,
                 clip_dc: bool = True,
                 rng_seed: int = 42):
        self.nps_root = nps_root
        self.target_std = float(target_noise_std_hu)
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.bh_strength = float(beam_hardening_strength)
        self.clip_dc = clip_dc

        # NPS 관련 파일 로드
        freqs_path = os.path.join(nps_root, "freqs.npy")
        radial_path = os.path.join(nps_root, "nps_radial.npy")
        stats_path = os.path.join(nps_root, "nps_stats.yaml")

        self.freqs = np.load(freqs_path)          # (F,)
        self.nps_radial = np.load(radial_path).copy()  # (F,) - copy for modification

        # DC 성분 clipping: freq=0의 NPS를 freq[1] 값으로 대체
        # 이유: DC(52842)가 고주파(~30) 대비 ~1700배로 과대하여 저주파 노이즈 과잉 발생
        if self.clip_dc and len(self.nps_radial) > 1:
            original_dc = self.nps_radial[0]
            self.nps_radial[0] = self.nps_radial[1]
            print(f"[NuclearNoiseModel] DC clipping: {original_dc:.0f} -> {self.nps_radial[0]:.0f}")

        if os.path.isfile(stats_path):
            with open(stats_path, "r") as f:
                yaml.safe_load(f)

        self.rng = np.random.RandomState(rng_seed)

    # ---------------------- 내부 유틸 ---------------------- #
    def _normalize_hu(self, hu):
        hu = np.clip(hu, self.hu_min, self.hu_max)
        return (hu - self.hu_min) / (self.hu_max - self.hu_min)

    def _denormalize_hu(self, x):
        return x * (self.hu_max - self.hu_min) + self.hu_min

    def _generate_structured_noise(self, shape):
        """
        NPS(radial)를 이용해서 white noise를 frequency-domain에서 shaping
        """
        H, W = shape
        # 1) white noise
        white = self.rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)

        # 2) frequency grid (cycles/pixel)
        fy = np.fft.fftfreq(H)  # (-0.5,0.5] 범위
        fx = np.fft.fftfreq(W)
        fx_grid, fy_grid = np.meshgrid(fx, fy)
        radius = np.sqrt(fx_grid ** 2 + fy_grid ** 2)  # 0 ~ sqrt(2)*0.5

        # 3) radial NPS를 현재 radius에 맞게 interpolation
        r_max = self.freqs.max()
        r = np.clip(radius, 0.0, r_max)
        nps_interp = np.interp(r.ravel(),
                               self.freqs,
                               self.nps_radial,
                               left=self.nps_radial[0],
                               right=self.nps_radial[-1]).reshape(H, W)

        # 4) amplitude = sqrt(NPS) 로 필터 만들고, FFT domain에서 곱
        amp = np.sqrt(nps_interp + 1e-8)
        white_fft = np.fft.fft2(white)
        shaped_fft = white_fft * amp
        shaped = np.fft.ifft2(shaped_fft).real.astype(np.float32)

        # 5) target std HU에 맞추기
        cur_std = float(shaped.std() + 1e-8)
        shaped = shaped * (self.target_std / cur_std)

        return shaped

    def _tissue_noise_scale(self, hu_slice):
        """
        조직 밀도(HU)에 따른 노이즈 강도 변조.

        물리적 근거 (BHA 논문 기반):
        - X선 감쇠는 물질 밀도에 비례 → 고밀도(bone) 통과 시 광자 수 감소
        - 광자 수 ∝ 1/attenuation → 노이즈 std ∝ sqrt(attenuation)
        - Bone(>300 HU) 근처: 노이즈 1.3~1.5배 증가
        - Air(<-500 HU): 노이즈 감소 (감쇠 거의 없음)
        - Soft tissue(-100~100 HU): 기준 (1.0)

        Returns:
            scale_map: (H, W) float32, 노이즈 배율 [0.7, 1.5]
        """
        # HU → 상대 감쇠 계수 (water=0 HU 기준 정규화)
        # 선형 감쇠: μ(HU) = μ_water × (1 + HU/1000)
        relative_atten = 1.0 + hu_slice / 1000.0
        relative_atten = np.clip(relative_atten, 0.05, 3.0)

        # 노이즈 ∝ sqrt(감쇠) (Poisson 통계)
        scale = np.sqrt(relative_atten).astype(np.float32)

        # 범위 제한: [0.7, 1.5]
        return np.clip(scale, 0.7, 1.5)

    def _add_beam_hardening(self, hu_slice):
        """
        물리 기반 beam hardening cupping artifact.

        X선이 물체를 통과할 때 저에너지 광자가 먼저 흡수되어
        빔이 "hardened"됨 → 중심부 HU가 실제보다 낮게 측정됨 (cupping).

        개선: 단순 radial이 아닌, body mask 기반 path length 추정.
        """
        if self.bh_strength <= 0.0:
            return hu_slice

        H, W = hu_slice.shape
        yy, xx = np.mgrid[0:H, 0:W]
        cy, cx = H / 2.0, W / 2.0
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_norm = r / (r.max() + 1e-8)

        # 조직이 있는 영역에서만 cupping 적용 (air 제외)
        body_region = (hu_slice > -500).astype(np.float32)

        # Cupping: 중심에서 더 강함 (path length가 길어서)
        # 2차 함수: 중심=최대, 가장자리=0
        cupping = (1.0 - r_norm ** 2) * body_region

        delta = -self.bh_strength * cupping * self.target_std
        return (hu_slice + delta.astype(np.float32)).astype(np.float32)

    # ---------------------- 공개 API ---------------------- #
    def add_noise(self, clean_hu: np.ndarray) -> np.ndarray:
        """
        입력: clean_hu (H,W) in HU
        출력: NPS 기반 structured noise가 추가된 noisy_hu

        물리 기반 개선:
        - 조직 밀도별 노이즈 강도 변조 (Poisson 통계 근사)
        - Bone 근처 노이즈 증가, Air 영역 노이즈 감소
        """
        H, W = clean_hu.shape

        # NPS 기반 structured noise
        structured = self._generate_structured_noise((H, W))

        # 조직 밀도에 따른 노이즈 강도 변조
        tissue_scale = self._tissue_noise_scale(clean_hu)
        structured = structured * tissue_scale

        noisy = clean_hu.astype(np.float32) + structured

        # beam-hardening: config에서 0이면 skip
        if self.bh_strength > 0:
            noisy = self._add_beam_hardening(noisy)

        return noisy.astype(np.float32)

    def add_artifact_aware_noise(self,
                                  clean_hu: np.ndarray,
                                  artifact_mask: np.ndarray,
                                  base_noise_hu: float = 35.0,
                                  artifact_noise_hu: float = 70.0) -> np.ndarray:
        """
        아티팩트 타겟 노이즈 주입 (현재 미사용).

        Args:
            clean_hu: (H, W) HU 이미지
            artifact_mask: (H, W) 아티팩트 마스크 [0, 1]
            base_noise_hu: 비아티팩트 영역 노이즈 std (HU)
            artifact_noise_hu: 아티팩트 영역 노이즈 std (HU)

        Returns:
            noisy_hu: NPS 구조 노이즈가 추가된 이미지
        """
        H, W = clean_hu.shape

        # NPS 구조 노이즈 생성 (표준화)
        base_noise = self._generate_structured_noise((H, W))
        base_noise = base_noise / (self.target_std + 1e-8)

        # 영역별 노이즈 강도
        artifact_mask = np.clip(artifact_mask, 0, 1).astype(np.float32)
        noise_strength = (1 - artifact_mask) * base_noise_hu + artifact_mask * artifact_noise_hu

        # NPS 노이즈만 적용 (white Gaussian 제거)
        shaped_noise = base_noise * noise_strength
        noisy = clean_hu.astype(np.float32) + shaped_noise

        # beam-hardening: config에서 0이면 skip
        if self.bh_strength > 0:
            noisy = self._add_beam_hardening(noisy)

        return noisy.astype(np.float32)
