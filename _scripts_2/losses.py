"""
losses.py - MA-HybridNet 손실 함수
Medical-Aware Loss + Gradient-Domain Texture Discriminator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple


# ============================================================
# Perceptual Loss (VGG Feature Matching)
# ============================================================

class PerceptualLoss(nn.Module):
    """
    VGG16의 중간 특징 맵을 활용하여
    영상의 고수준 구조적 유사도를 비교한다.
    장기(Organ)의 형태가 왜곡되는 것을 방지.

    CT는 1채널이므로 3채널로 복제하여 VGG에 입력한다.
    """

    def __init__(self, feature_layers: Optional[list] = None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        if feature_layers is None:
            feature_layers = [3, 8, 15, 22]    # relu1_2, relu2_2, relu3_3, relu4_3

        self.feature_layers = feature_layers
        max_layer = max(feature_layers) + 1
        self.features = nn.Sequential(*list(vgg.features.children())[:max_layer])

        # VGG 가중치 동결
        for param in self.features.parameters():
            param.requires_grad = False

        # ImageNet 정규화 (float32 정밀도 유지)
        self.register_buffer('mean',
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] → ImageNet 정규화."""
        x = (x + 1.0) / 2.0                        # [0, 1]
        x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        x = (x - self.mean) / self.std
        return x

    def _extract_features(self, x: torch.Tensor) -> list:
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)

        pred_feats = self._extract_features(pred_norm)
        target_feats = self._extract_features(target_norm)

        loss = torch.tensor(0.0, device=pred.device)
        for pf, tf in zip(pred_feats, target_feats):
            loss = loss + F.l1_loss(pf, tf.detach())

        return loss / len(pred_feats)


# ============================================================
# SSIM Loss (Structural Similarity)
# ============================================================

class SSIMLoss(nn.Module):
    """
    구조적 유사도(SSIM)를 손실 함수로 사용하여
    영상의 구조적 정보(밝기, 대비, 구조)를 보존한다.
    장기(Organ) 경계와 해부학적 구조 왜곡 방지.
    float32 정밀도 유지를 위해 순수 텐서 연산으로 구현.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2

        # 가우시안 윈도우 생성
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        self.register_buffer('window',
                             window_2d.unsqueeze(0).unsqueeze(0))  # (1, 1, K, K)
        self.pad = window_size // 2

    def _ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # AMP 호환: window를 input과 같은 dtype/device로 변환
        w = self.window.to(dtype=x.dtype, device=x.device)
        mu_x = F.conv2d(x, w, padding=self.pad)
        mu_y = F.conv2d(y, w, padding=self.pad)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.conv2d(x ** 2, w, padding=self.pad) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, w, padding=self.pad) - mu_y_sq
        sigma_xy = F.conv2d(x * y, w, padding=self.pad) - mu_xy

        ssim = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
               ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))
        return ssim

    def forward(self, pred: torch.Tensor, target: torch.Tensor
                ) -> torch.Tensor:
        """1 - SSIM을 손실로 사용 (SSIM이 1일수록 유사 → 손실 0)."""
        ssim_map = self._ssim_map(pred, target)
        return 1.0 - ssim_map.mean()


# ============================================================
# Gradient-Domain Texture Discriminator
# ============================================================

def compute_sobel_2ch(x: torch.Tensor) -> torch.Tensor:
    """
    Sobel gradient를 2채널로 반환.
    Input:  (B, 1, H, W) normalized image
    Output: (B, 2, H, W) [grad_x, grad_y]

    Gradient domain에서 작동하면 HU 절대값(조영 효과) 차이를 무시하고
    텍스처 패턴(edge, grain)만 비교할 수 있음.
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.cat([gx, gy], dim=1)  # (B, 2, H, W)


class GradientTextureDiscriminator(nn.Module):
    """
    PatchGAN Discriminator operating in gradient (Sobel) domain.

    CE-CT의 텍스처 분포를 학습하여, 모델 출력이 "진짜 CT 텍스처"를 갖도록 유도.
    Gradient domain에서 작동하므로 조영 효과(HU 차이)는 자동으로 무시됨.

    Input:  Sobel gradient (B, 2, H, W)
    Output: PatchGAN prediction grid (B, 1, H', W')
    Receptive field: ~70x70
    """

    def __init__(self, in_channels: int = 2, ndf: int = 64):
        super().__init__()

        # 4-layer PatchGAN: 2→64→128→256→1
        self.model = nn.Sequential(
            # Layer 1: no norm
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: stride 1
            nn.Conv2d(ndf * 4, ndf * 4, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Output: 1-channel prediction
            nn.Conv2d(ndf * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, H, W) Sobel gradient
        Returns:
            (B, 1, H', W') patch-level real/fake predictions
        """
        return self.model(x)


# ============================================================
# Medical-Aware Losses (의료 영상 특화 손실 함수)
# ============================================================

class NPSLoss(nn.Module):
    """
    Noise Power Spectrum (NPS) Loss.
    Denoised 이미지의 노이즈 텍스처가 실제 CT 노이즈 특성(NPS)과 유사하도록 유도.

    핵심: 고주파 노이즈는 제거하되, 현실적인 노이즈 텍스처 보존.
    병변과 노이즈를 구분하는 핵심은 주파수 특성.
    """

    def __init__(self, num_freq_bins: int = 32):
        super().__init__()
        self.num_bins = num_freq_bins

    def _compute_radial_nps(self, img: torch.Tensor) -> torch.Tensor:
        """
        2D FFT → Radial average로 NPS 계산.

        Args:
            img: (B, 1, H, W) [-1, 1] normalized image
        Returns:
            nps_radial: (B, num_bins) Radial NPS profile
        """
        B, C, H, W = img.shape

        # 2D FFT (DC shift)
        img_fft = torch.fft.fft2(img.squeeze(1))  # (B, H, W)
        img_fft = torch.fft.fftshift(img_fft)

        # Power spectrum
        power = torch.abs(img_fft) ** 2  # (B, H, W)

        # Radial binning
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=img.device) - cy
        x = torch.arange(W, device=img.device) - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(yy ** 2 + xx ** 2)  # (H, W)

        max_radius = min(cy, cx)
        radius_normalized = radius / max_radius  # [0, sqrt(2)]

        # Binning
        nps_radial = []
        for b in range(B):
            bins = []
            for i in range(self.num_bins):
                r_min = i / self.num_bins
                r_max = (i + 1) / self.num_bins
                mask = (radius_normalized >= r_min) & (radius_normalized < r_max)
                if mask.sum() > 0:
                    bins.append(power[b][mask].mean())
                else:
                    bins.append(torch.tensor(0.0, device=img.device))
            nps_radial.append(torch.stack(bins))

        return torch.stack(nps_radial)  # (B, num_bins)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        NPS similarity loss.

        Args:
            pred: (B, 1, H, W) Denoised output
            target: (B, 1, H, W) Target (NC-CT original with noise)
        """
        nps_pred = self._compute_radial_nps(pred)
        nps_target = self._compute_radial_nps(target)

        # MSE on NPS profiles
        loss = F.mse_loss(nps_pred, nps_target)

        return loss


class MultiScaleTextureLoss(nn.Module):
    """
    Multi-Scale Texture Preservation Loss.

    해부학적 텍스처(장기 내부 패턴, 혈관, 미세조직)를 다중 스케일에서 보존.
    Gabor filter bank으로 방향성 텍스처 추출 → 유사도 측정.
    """

    def __init__(self, scales: list = [1, 2, 4], num_orientations: int = 4):
        super().__init__()
        self.scales = scales
        self.num_orientations = num_orientations

        # Gabor filter bank (방향성 텍스처)
        self.gabor_kernels = self._create_gabor_filters()

    def _create_gabor_filters(self) -> nn.ParameterList:
        """
        Gabor 필터 생성 (4방향 × 3스케일 = 12 filters).

        Gabor: 특정 방향의 텍스처 감지 (혈관, 조직 패턴)
        """
        kernels = []
        kernel_size = 15
        sigma = 3.0

        for scale in self.scales:
            for theta in torch.linspace(0, torch.pi, self.num_orientations + 1)[:-1]:
                # Gabor kernel 생성
                lambd = kernel_size / (2.0 * scale)  # Wavelength
                gamma = 0.5  # Aspect ratio

                y, x = torch.meshgrid(
                    torch.linspace(-kernel_size//2, kernel_size//2, kernel_size),
                    torch.linspace(-kernel_size//2, kernel_size//2, kernel_size),
                    indexing='ij'
                )

                # Rotation
                x_theta = x * torch.cos(theta) + y * torch.sin(theta)
                y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

                # Gabor function
                gaussian = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
                sinusoid = torch.cos(2 * torch.pi * x_theta / lambd)
                gabor = gaussian * sinusoid

                # Normalize
                gabor = gabor / (gabor.abs().sum() + 1e-8)

                kernels.append(gabor.view(1, 1, kernel_size, kernel_size))

        # Register as buffers (non-trainable)
        kernel_tensor = torch.cat(kernels, dim=0)  # (12, 1, 15, 15)
        self.register_buffer('gabor_bank', kernel_tensor)

        return kernels

    def _extract_texture_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Gabor filter bank으로 텍스처 특징 추출.

        Args:
            img: (B, 1, H, W)
        Returns:
            features: (B, 12, H, W) - 12개 필터 응답
        """
        B = img.shape[0]
        gabor_bank = self.gabor_bank.to(dtype=img.dtype, device=img.device)
        features = F.conv2d(img, gabor_bank, padding=7)  # (B, 12, H, W)
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Texture similarity loss.

        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)
            weight_mask: (B, 1, H, W) Optional - 조직별 가중치 (water 영역 강조 등)
        """
        feat_pred = self._extract_texture_features(pred)
        feat_target = self._extract_texture_features(target)

        # L1 distance on texture responses
        texture_diff = torch.abs(feat_pred - feat_target)

        if weight_mask is not None:
            # 특정 영역(예: 저음영 water 영역) 강조
            weight_mask = F.interpolate(weight_mask, size=texture_diff.shape[2:],
                                         mode='bilinear', align_corners=False)
            texture_diff = texture_diff * (1.0 + weight_mask)

        return texture_diff.mean()


class EdgePreservingLoss(nn.Module):
    """
    Edge-Preserving Loss.

    장기 경계, 혈관 경계 보존.
    Sobel gradient magnitude 비교 → 경계 왜곡 방지.
    """

    def __init__(self):
        super().__init__()

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _compute_gradient(self, img: torch.Tensor) -> torch.Tensor:
        """Gradient magnitude (edge strength)."""
        sobel_x = self.sobel_x.to(dtype=img.dtype, device=img.device)
        sobel_y = self.sobel_y.to(dtype=img.dtype, device=img.device)
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return grad_mag

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Edge preservation loss.

        경계가 강한 영역(장기, 혈관)에서 gradient 유사도 강제.
        """
        grad_pred = self._compute_gradient(pred)
        grad_target = self._compute_gradient(target)

        # L1 loss on gradients
        return F.l1_loss(grad_pred, grad_target)


class LowFrequencyLoss(nn.Module):
    """
    Low-Frequency Preservation Loss.

    저주파 성분 = 구조, 저음영 영역 (위액, 방광, 낭종 등).
    Gaussian blur로 저주파 추출 → 보존.

    핵심: 저음영 병변/물이 노이즈로 오인되어 제거되는 것 방지.
    """

    def __init__(self, sigma: float = 3.0):
        super().__init__()
        self.sigma = sigma

        # Gaussian kernel 생성
        kernel_size = int(sigma * 4) | 1  # 홀수
        if kernel_size < 3:
            kernel_size = 3

        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()

        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

        self.register_buffer('gaussian', gauss_2d)
        self.pad = kernel_size // 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Low-frequency component preservation.

        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)
            weight_mask: (B, 1, H, W) - 저음영 영역(water) 가중치
        """
        # Gaussian blur = 저주파 추출
        gaussian = self.gaussian.to(dtype=pred.dtype, device=pred.device)
        lowfreq_pred = F.conv2d(pred, gaussian, padding=self.pad)
        lowfreq_target = F.conv2d(target, gaussian, padding=self.pad)

        # L1 loss on low-frequency
        loss = torch.abs(lowfreq_pred - lowfreq_target)

        if weight_mask is not None:
            # 저음영 영역(water, 병변 후보) 강조
            loss = loss * (1.0 + weight_mask)

        return loss.mean()


class ArtifactSuppressionLoss(nn.Module):
    """
    Artifact Suppression Loss.

    팔 뼈로 인한 beam hardening, streak artifact 억제.
    Bone 영역 근처에서 radial pattern 감지 → 억제.
    """

    def __init__(self):
        super().__init__()

    def _detect_radial_artifact(self, img: torch.Tensor,
                                 bone_mask: torch.Tensor) -> torch.Tensor:
        """
        Radial artifact detection.

        Bone 영역 중심에서 방사형 패턴 감지 (streak artifact).
        """
        B, C, H, W = img.shape

        # Bone mask erosion (중심만 남김)
        bone_center = F.avg_pool2d(bone_mask, kernel_size=5, stride=1, padding=2)
        bone_center = (bone_center > 0.5).float()

        # Radial gradient (중심에서 바깥 방향)
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=img.device, dtype=torch.float32) - cy
        x = torch.arange(W, device=img.device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Radial direction
        radius = torch.sqrt(yy ** 2 + xx ** 2 + 1e-6)
        radial_y = yy / radius
        radial_x = xx / radius

        # Image gradient (AMP 호환)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)

        # Radial alignment (dot product)
        radial_alignment = (grad_x.squeeze(1) * radial_x +
                             grad_y.squeeze(1) * radial_y).unsqueeze(1)

        # Artifact score (high radial alignment = artifact)
        artifact_score = torch.abs(radial_alignment) * bone_center

        return artifact_score

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                bone_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Artifact suppression in bone regions.

        Args:
            pred: (B, 1, H, W)
            target: (B, 1, H, W)
            bone_mask: (B, 1, H, W) - Bone region mask
        """
        if bone_mask is None:
            return torch.tensor(0.0, device=pred.device)

        # Detect radial artifacts in prediction
        artifact_pred = self._detect_radial_artifact(pred, bone_mask)

        # Penalize strong radial patterns (streak artifacts)
        # Target should have less artifact
        artifact_target = self._detect_radial_artifact(target, bone_mask)

        # Encourage pred to have less artifact than target
        return F.relu(artifact_pred - artifact_target).mean()


class UniformityLoss(nn.Module):
    """
    Uniformity Loss.

    균일 조직 내 HU 편차를 줄여 beam hardening artifact의
    국소적 HU 단차(patchy)를 억제.

    원리: 로컬 윈도우 내 분산(pred) > 분산(target)이면 페널티.
    균일해야 할 영역(soft tissue)에서 pred가 target보다
    더 불균일하지 않도록 강제.

    Ref: BHA 보정 논문의 Uniformity 지표 개념 응용.
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2
        kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
        self.register_buffer('kernel', kernel)

    def _local_variance(self, x: torch.Tensor) -> torch.Tensor:
        """로컬 분산 = E[x^2] - E[x]^2"""
        k = self.kernel.to(dtype=x.dtype, device=x.device)
        mu = F.conv2d(x, k, padding=self.pad)
        mu_sq = F.conv2d(x ** 2, k, padding=self.pad)
        return F.relu(mu_sq - mu ** 2)  # 수치 안정성

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        pred의 로컬 분산이 target보다 큰 경우만 페널티.

        Args:
            weight_mask: soft tissue 영역 가중치 (bone 제외)
        """
        var_pred = self._local_variance(pred)
        var_target = self._local_variance(target)

        # pred가 target보다 불균일한 경우만 페널티
        excess = F.relu(var_pred - var_target)

        if weight_mask is not None:
            excess = excess * weight_mask

        return excess.mean()


class MedicalAwareLoss(nn.Module):
    """
    통합 의료 영상 특화 손실 함수.

    Components:
        1. L1: Pixel-wise fidelity
        2. SSIM: Structural similarity
        3. Edge: 경계 보존 (장기, 혈관)
        4. Low-freq: 저주파 보존 (저음영 병변/물)
        5. NPS: 노이즈 텍스처 현실성
        6. Texture: 해부학적 텍스처 보존
        7. Artifact: 뼈 아티팩트 억제
        8. Uniformity: 균일 조직 내 HU 편차 억제
    """

    def __init__(self,
                 lambda_l1: float = 1.0,
                 lambda_ssim: float = 2.0,
                 lambda_edge: float = 1.5,
                 lambda_lowfreq: float = 2.0,
                 lambda_nps: float = 0.5,
                 lambda_texture: float = 1.0,
                 lambda_artifact: float = 1.0,
                 lambda_noise_residual: float = 2.0,
                 lambda_uniformity: float = 0.5,
                 inpaint_trust: float = 0.3):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_lowfreq = lambda_lowfreq
        self.lambda_nps = lambda_nps
        self.lambda_texture = lambda_texture
        self.lambda_artifact = lambda_artifact
        self.lambda_noise_residual = lambda_noise_residual
        self.lambda_uniformity = lambda_uniformity
        self.inpaint_trust = inpaint_trust

        # Loss modules
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgePreservingLoss()
        self.lowfreq_loss = LowFrequencyLoss(sigma=3.0)
        self.nps_loss = NPSLoss(num_freq_bins=32)
        self.texture_loss = MultiScaleTextureLoss(scales=[1, 2, 4])
        self.artifact_loss = ArtifactSuppressionLoss()
        self.uniformity_loss = UniformityLoss(window_size=11)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                input_noisy: Optional[torch.Tensor] = None,
                noise_pred: Optional[torch.Tensor] = None,
                water_mask: Optional[torch.Tensor] = None,
                bone_mask: Optional[torch.Tensor] = None,
                action_masks: Optional[dict] = None,
                inpaint_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        통합 손실 계산 (Residual Learning + Action-Guided Weighting).

        Args:
            pred: (B, 1, H, W) - Denoised output (input - noise_pred)
            target: (B, 1, H, W) - Target (NC-CT original for N2N)
            input_noisy: (B, 1, H, W) - Noisy input (for noise residual loss)
            noise_pred: (B, 1, H, W) - Predicted noise
            water_mask: (B, 1, H, W) - 저음영 영역 mask (HU-based, from model)
            bone_mask: (B, 1, H, W) - Bone region mask (HU-based, from model)
            action_masks: dict with pre-computed spatial guidance masks:
                'artifact':  (B, 1, H, W) [0,1] → loss weighting (제거 강화)
                'fluid':     (B, 1, H, W) [0,1] → lowfreq/texture 가중 (보존 강화)
                'structure': (B, 1, H, W) [0,1] → SSIM/edge 가중 (경계 보존 강화)
            inpaint_mask: (B, 1, H, W) [0,1] → |raw-processed| diff 기반
                실제 수정된 픽셀. L1/noise_residual trust 감소에 사용.
                artifact_mask와 분리: artifact_mask는 loss weighting용,
                inpaint_mask는 target 신뢰도용.

        Returns:
            total_loss: Scalar tensor
            loss_dict: Dict of individual losses
        """
        losses = {}

        # Action mask 분리 (없으면 None)
        art_w = flu_w = str_w = None
        if action_masks is not None:
            art_w = action_masks.get('artifact')    # (B, 1, H, W) [0, 1]
            flu_w = action_masks.get('fluid')
            str_w = action_masks.get('structure')

        # Inpaint 영역에서 action mask 신뢰도 감소
        # 이유: action mask는 raw CT(artifact 오염)에서 계산됨
        # → inpaint된 곳에서는 tissue 오분류, streak=structure 등 mask 자체가 부정확
        # → 해당 영역에서 action mask 영향력을 줄여 모델이 오도되지 않게 함
        if inpaint_mask is not None:
            mask_reliability = 1.0 - inpaint_mask * 0.8  # inpaint=1 → 0.2배로 감쇠
            if art_w is not None:
                art_w = art_w * mask_reliability
            if str_w is not None:
                str_w = str_w * mask_reliability
            # flu_w는 감쇠하지 않음: fluid는 artifact와 무관하게 보존 필요

        # ----------------------------------------------------------
        # 1. L1 Loss (inpaint 영역 신뢰도 감소)
        # ----------------------------------------------------------
        # inpaint_mask (|raw-processed| diff): 실제 수정된 픽셀 기반
        # → artifact_mask보다 정밀 (tissue 오분류 무관)
        # → inpaint된 곳: L1 trust 감소 (target이 interpolation이므로)
        # → 비-inpaint: 기본 trust (target = 원본 그대로)
        if inpaint_mask is not None:
            l1_map = torch.abs(pred - target)
            # 연속 가중치: inpaint_mask 0→trust 1.0, inpaint_mask 1→trust inpaint_trust
            l1_weight = 1.0 - inpaint_mask * (1.0 - self.inpaint_trust)
            losses['l1'] = (l1_map * l1_weight).mean()
        else:
            losses['l1'] = self.l1_loss(pred, target)

        # ----------------------------------------------------------
        # 2. Noise Residual Loss (inpaint 영역 신뢰도 감소)
        # ----------------------------------------------------------
        if input_noisy is not None and noise_pred is not None:
            noise_gt = input_noisy - target
            if inpaint_mask is not None:
                nr_map = torch.abs(noise_pred - noise_gt)
                nr_weight = 1.0 - inpaint_mask * (1.0 - self.inpaint_trust)
                losses['noise_residual'] = (nr_map * nr_weight).mean()
            else:
                losses['noise_residual'] = F.l1_loss(noise_pred, noise_gt)
        else:
            losses['noise_residual'] = torch.tensor(0.0, device=pred.device)

        # ----------------------------------------------------------
        # 3. SSIM Loss (structure 영역 가중: "경계 살려")
        # ----------------------------------------------------------
        if str_w is not None:
            ssim_map = self.ssim_loss._ssim_map(pred, target)
            ssim_diff = 1.0 - ssim_map
            ssim_weight = 1.0 + str_w * 2.0
            losses['ssim'] = (ssim_diff * ssim_weight).mean()
        else:
            losses['ssim'] = self.ssim_loss(pred, target)

        # ----------------------------------------------------------
        # 4. Edge Loss (structure 영역 가중)
        # ----------------------------------------------------------
        if str_w is not None:
            grad_pred = self.edge_loss._compute_gradient(pred)
            grad_target = self.edge_loss._compute_gradient(target)
            edge_diff = torch.abs(grad_pred - grad_target)
            edge_weight = 1.0 + str_w * 2.0
            losses['edge'] = (edge_diff * edge_weight).mean()
        else:
            losses['edge'] = self.edge_loss(pred, target)

        # ----------------------------------------------------------
        # 5. Low-frequency Loss (fluid 영역 가중: "여기는 절대 보존")
        # ----------------------------------------------------------
        # fluid mask와 model의 water_mask 중 더 넓은 범위 사용
        if flu_w is not None and water_mask is not None:
            combined_fluid = torch.max(flu_w, water_mask)
        elif flu_w is not None:
            combined_fluid = flu_w
        else:
            combined_fluid = water_mask
        losses['lowfreq'] = self.lowfreq_loss(pred, target, weight_mask=combined_fluid)

        # ----------------------------------------------------------
        # 6. NPS Loss (비활성화)
        # ----------------------------------------------------------
        losses['nps'] = torch.tensor(0.0, device=pred.device)

        # ----------------------------------------------------------
        # 7. Texture Loss (fluid 영역 가중)
        # ----------------------------------------------------------
        losses['texture'] = self.texture_loss(pred, target, weight_mask=combined_fluid)

        # ----------------------------------------------------------
        # 8. Artifact Loss (bone 영역)
        # ----------------------------------------------------------
        losses['artifact'] = self.artifact_loss(pred, target, bone_mask=bone_mask)

        # ----------------------------------------------------------
        # 9. Uniformity Loss (soft tissue 균일도)
        # ----------------------------------------------------------
        # bone 제외한 soft tissue 영역에서만 (bone은 원래 불균일)
        soft_tissue_mask = (1.0 - bone_mask) if bone_mask is not None else None
        losses['uniformity'] = self.uniformity_loss(pred, target, weight_mask=soft_tissue_mask)

        # Total
        total = (self.lambda_l1 * losses['l1'] +
                 self.lambda_noise_residual * losses['noise_residual'] +
                 self.lambda_ssim * losses['ssim'] +
                 self.lambda_edge * losses['edge'] +
                 self.lambda_lowfreq * losses['lowfreq'] +
                 self.lambda_texture * losses['texture'] +
                 self.lambda_artifact * losses['artifact'] +
                 self.lambda_uniformity * losses['uniformity'])

        # Convert to items for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items()}
        loss_dict['total'] = total.item()

        return total, loss_dict
