"""
losses.py - Streak Removal 특화 손실 함수

변경 이력:
- v1: MedicalAwareLoss (디노이징용, 8개 loss component)
- v2: StreakRemovalLoss (artifact 제거 특화)
  - 제거: noise_residual, NPS, lowfreq (노이즈 관련)
  - 추가: PerceptualLoss (VGG feature matching), StreakWeightedL1
  - 유지: SSIM, Edge, Texture, Artifact, Uniformity
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
    VGG16 중간 특징 맵 기반 지각적 유사도 손실.

    Over-smoothing 방지: L1/SSIM만으로는 평균적으로 올바른 답에 수렴하여
    결과가 흐려짐. VGG feature 공간에서 비교하면 고수준 구조/텍스처가 보존됨.

    CT는 1채널이므로 3채널로 복제하여 VGG에 입력.
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

        # ImageNet 정규화
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
    구조적 유사도(SSIM) 손실.
    장기 경계와 해부학적 구조 보존.
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5,
                 C1: float = 0.01 ** 2, C2: float = 0.03 ** 2):
        super().__init__()
        self.C1 = C1
        self.C2 = C2

        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        g = g / g.sum()
        window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
        self.register_buffer('window',
                             window_2d.unsqueeze(0).unsqueeze(0))
        self.pad = window_size // 2

    def _ssim_map(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
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
        ssim_map = self._ssim_map(pred, target)
        return 1.0 - ssim_map.mean()


# ============================================================
# Gradient-Domain Texture Discriminator
# ============================================================

def compute_sobel_2ch(x: torch.Tensor) -> torch.Tensor:
    """
    Sobel gradient를 2채널로 반환.
    Input:  (B, 1, H, W)
    Output: (B, 2, H, W) [grad_x, grad_y]
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.cat([gx, gy], dim=1)


class GradientTextureDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (gradient domain).
    CE-CT 텍스처 분포 학습 → 모델 출력이 진짜 CT 텍스처를 갖도록 유도.
    """

    def __init__(self, in_channels: int = 2, ndf: int = 64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 4, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================
# Edge-Preserving Loss
# ============================================================

class EdgePreservingLoss(nn.Module):
    """장기/혈관 경계 보존. Sobel gradient magnitude 비교."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _compute_gradient(self, img: torch.Tensor) -> torch.Tensor:
        sobel_x = self.sobel_x.to(dtype=img.dtype, device=img.device)
        sobel_y = self.sobel_y.to(dtype=img.dtype, device=img.device)
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grad_pred = self._compute_gradient(pred)
        grad_target = self._compute_gradient(target)
        return F.l1_loss(grad_pred, grad_target)


# ============================================================
# Multi-Scale Texture Loss (Gabor filter bank)
# ============================================================

class MultiScaleTextureLoss(nn.Module):
    """다중 스케일 Gabor filter bank 기반 조직 텍스처 보존."""

    def __init__(self, scales: list = [1, 2, 4], num_orientations: int = 4):
        super().__init__()
        self.scales = scales
        self.num_orientations = num_orientations
        self._create_gabor_filters()

    def _create_gabor_filters(self):
        kernels = []
        kernel_size = 15
        sigma = 3.0

        for scale in self.scales:
            for theta in torch.linspace(0, torch.pi, self.num_orientations + 1)[:-1]:
                lambd = kernel_size / (2.0 * scale)
                gamma = 0.5

                y, x = torch.meshgrid(
                    torch.linspace(-kernel_size//2, kernel_size//2, kernel_size),
                    torch.linspace(-kernel_size//2, kernel_size//2, kernel_size),
                    indexing='ij'
                )

                x_theta = x * torch.cos(theta) + y * torch.sin(theta)
                y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

                gaussian = torch.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
                sinusoid = torch.cos(2 * torch.pi * x_theta / lambd)
                gabor = gaussian * sinusoid
                gabor = gabor / (gabor.abs().sum() + 1e-8)

                kernels.append(gabor.view(1, 1, kernel_size, kernel_size))

        kernel_tensor = torch.cat(kernels, dim=0)
        self.register_buffer('gabor_bank', kernel_tensor)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        gabor_bank = self.gabor_bank.to(dtype=pred.dtype, device=pred.device)
        feat_pred = F.conv2d(pred, gabor_bank, padding=7)
        feat_target = F.conv2d(target, gabor_bank, padding=7)

        texture_diff = torch.abs(feat_pred - feat_target)

        if weight_mask is not None:
            weight_mask = F.interpolate(weight_mask, size=texture_diff.shape[2:],
                                         mode='bilinear', align_corners=False)
            texture_diff = texture_diff * (1.0 + weight_mask)

        return texture_diff.mean()


# ============================================================
# Artifact Suppression Loss (Radial streak pattern)
# ============================================================

class ArtifactSuppressionLoss(nn.Module):
    """Bone 영역 중심 방사형 streak artifact 억제."""

    def __init__(self):
        super().__init__()

    def _detect_radial_artifact(self, img: torch.Tensor,
                                 bone_mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape

        bone_center = F.avg_pool2d(bone_mask, kernel_size=5, stride=1, padding=2)
        bone_center = (bone_center > 0.5).float()

        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=img.device, dtype=torch.float32) - cy
        x = torch.arange(W, device=img.device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        radius = torch.sqrt(yy ** 2 + xx ** 2 + 1e-6)
        radial_y = yy / radius
        radial_x = xx / radius

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)

        radial_alignment = (grad_x.squeeze(1) * radial_x +
                             grad_y.squeeze(1) * radial_y).unsqueeze(1)

        artifact_score = torch.abs(radial_alignment) * bone_center
        return artifact_score

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                bone_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if bone_mask is None:
            return torch.tensor(0.0, device=pred.device)

        artifact_pred = self._detect_radial_artifact(pred, bone_mask)
        artifact_target = self._detect_radial_artifact(target, bone_mask)

        return F.relu(artifact_pred - artifact_target).mean()


# ============================================================
# Uniformity Loss
# ============================================================

class UniformityLoss(nn.Module):
    """
    균일 조직 내 HU 편차 억제.
    Streak artifact이 만드는 patchy HU 단차를 직접 페널티.
    pred의 로컬 분산 > target의 로컬 분산이면 페널티.
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2
        kernel = torch.ones(1, 1, window_size, window_size) / (window_size ** 2)
        self.register_buffer('kernel', kernel)

    def _local_variance(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel.to(dtype=x.dtype, device=x.device)
        mu = F.conv2d(x, k, padding=self.pad)
        mu_sq = F.conv2d(x ** 2, k, padding=self.pad)
        return F.relu(mu_sq - mu ** 2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weight_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        var_pred = self._local_variance(pred)
        var_target = self._local_variance(target)

        excess = F.relu(var_pred - var_target)

        if weight_mask is not None:
            excess = excess * weight_mask

        return excess.mean()


# ============================================================
# Streak-Weighted L1 Loss (NEW - artifact 제거 특화)
# ============================================================

class StreakWeightedL1Loss(nn.Module):
    """
    Streak Map 가중 L1 Loss.

    핵심 아이디어:
    - streak_map이 강한 영역 = artifact가 심한 곳
    - 해당 영역에서 L1 loss를 amplify → 모델이 streak 제거에 집중
    - streak_map이 약한 영역 = 정상 조직 → 기본 L1만 적용

    가중 공식:
        weight = 1.0 + streak_boost * |streak_map|
        → streak=0: weight=1.0 (기본)
        → streak=±1: weight=1.0+streak_boost (최대)
    """

    def __init__(self, streak_boost: float = 3.0):
        super().__init__()
        self.streak_boost = streak_boost

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                streak_map: Optional[torch.Tensor] = None,
                inpaint_mask: Optional[torch.Tensor] = None,
                inpaint_trust: float = 0.3) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) denoised output
            target: (B, 1, H, W) streak-corrected target
            streak_map: (B, 1, H, W) normalized streak intensity [-1, 1]
            inpaint_mask: (B, 1, H, W) [0, 1] inpaint된 픽셀
            inpaint_trust: inpaint 영역 신뢰도 (0~1)
        """
        l1_map = torch.abs(pred - target)

        # Streak 영역 가중
        if streak_map is not None:
            streak_weight = 1.0 + self.streak_boost * torch.abs(streak_map)
            l1_map = l1_map * streak_weight

        # Inpaint 영역 신뢰도 감소
        if inpaint_mask is not None:
            trust_weight = 1.0 - inpaint_mask * (1.0 - inpaint_trust)
            l1_map = l1_map * trust_weight

        return l1_map.mean()


# ============================================================
# Streak Removal Loss (통합)
# ============================================================

class StreakRemovalLoss(nn.Module):
    """
    Streak Removal 특화 통합 손실 함수.

    디노이징용 MedicalAwareLoss에서 변경:
    - 제거: noise_residual (노이즈 예측), NPS (노이즈 텍스처), lowfreq (streak 보존 위험)
    - 추가: PerceptualLoss (VGG feature, over-smoothing 방지)
    - 추가: StreakWeightedL1 (streak 영역 가중 L1)
    - 유지: SSIM, Edge, Texture, Artifact, Uniformity

    Components:
        1. streak_l1: streak map 가중 L1 (artifact 영역 집중)
        2. perceptual: VGG feature matching (over-smoothing 방지, 시각적 품질 핵심)
        3. ssim: 구조 보존
        4. edge: 장기/혈관 경계 보존
        5. texture: Gabor 기반 조직 텍스처 보존
        6. artifact: 방사형 streak 억제
        7. uniformity: 균일 조직 HU 편차 억제
    """

    def __init__(self,
                 lambda_streak_l1: float = 2.0,
                 lambda_perceptual: float = 1.0,
                 lambda_ssim: float = 1.0,
                 lambda_edge: float = 1.0,
                 lambda_texture: float = 0.5,
                 lambda_artifact: float = 2.0,
                 lambda_uniformity: float = 1.0,
                 inpaint_trust: float = 0.3,
                 streak_boost: float = 3.0):
        super().__init__()

        self.lambda_streak_l1 = lambda_streak_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_texture = lambda_texture
        self.lambda_artifact = lambda_artifact
        self.lambda_uniformity = lambda_uniformity
        self.inpaint_trust = inpaint_trust

        # Loss modules
        self.streak_l1_loss = StreakWeightedL1Loss(streak_boost=streak_boost)
        self.perceptual_loss = PerceptualLoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgePreservingLoss()
        self.texture_loss = MultiScaleTextureLoss(scales=[1, 2, 4])
        self.artifact_loss = ArtifactSuppressionLoss()
        self.uniformity_loss = UniformityLoss(window_size=11)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                bone_mask: Optional[torch.Tensor] = None,
                streak_map: Optional[torch.Tensor] = None,
                action_masks: Optional[dict] = None,
                inpaint_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Streak removal 통합 손실 계산.

        Args:
            pred: (B, 1, H, W) denoised output [-1, 1]
            target: (B, 1, H, W) streak-corrected target [-1, 1]
            bone_mask: (B, 1, H, W) bone region mask (from model)
            streak_map: (B, 1, H, W) normalized streak intensity [-1, 1]
            action_masks: dict with 'artifact', 'fluid', 'structure' [0, 1]
            inpaint_mask: (B, 1, H, W) [0, 1] inpaint된 픽셀

        Returns:
            total_loss: scalar
            loss_dict: 개별 loss 값 (logging용)
        """
        losses = {}

        # Action mask 분리
        art_w = str_w = None
        if action_masks is not None:
            art_w = action_masks.get('artifact')
            str_w = action_masks.get('structure')

        # Inpaint 영역에서 action mask 신뢰도 감소
        if inpaint_mask is not None:
            mask_reliability = 1.0 - inpaint_mask * 0.8
            if art_w is not None:
                art_w = art_w * mask_reliability
            if str_w is not None:
                str_w = str_w * mask_reliability

        # ----------------------------------------------------------
        # 1. Streak-Weighted L1 (핵심: streak 영역 가중)
        # ----------------------------------------------------------
        losses['streak_l1'] = self.streak_l1_loss(
            pred, target,
            streak_map=streak_map,
            inpaint_mask=inpaint_mask,
            inpaint_trust=self.inpaint_trust,
        )

        # ----------------------------------------------------------
        # 2. Perceptual Loss (VGG feature matching)
        # ----------------------------------------------------------
        losses['perceptual'] = self.perceptual_loss(pred, target)

        # ----------------------------------------------------------
        # 3. SSIM Loss (structure 영역 가중)
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
        # 5. Texture Loss (조직 텍스처 보존)
        # ----------------------------------------------------------
        losses['texture'] = self.texture_loss(pred, target)

        # ----------------------------------------------------------
        # 6. Artifact Loss (bone 근처 방사형 패턴 억제)
        # ----------------------------------------------------------
        losses['artifact'] = self.artifact_loss(pred, target, bone_mask=bone_mask)

        # ----------------------------------------------------------
        # 7. Uniformity Loss (streak으로 인한 HU 편차 억제)
        # ----------------------------------------------------------
        soft_tissue_mask = (1.0 - bone_mask) if bone_mask is not None else None
        losses['uniformity'] = self.uniformity_loss(pred, target, weight_mask=soft_tissue_mask)

        # ----------------------------------------------------------
        # Total
        # ----------------------------------------------------------
        total = (self.lambda_streak_l1 * losses['streak_l1'] +
                 self.lambda_perceptual * losses['perceptual'] +
                 self.lambda_ssim * losses['ssim'] +
                 self.lambda_edge * losses['edge'] +
                 self.lambda_texture * losses['texture'] +
                 self.lambda_artifact * losses['artifact'] +
                 self.lambda_uniformity * losses['uniformity'])

        # Logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v
                     for k, v in losses.items()}
        loss_dict['total'] = total.item()

        return total, loss_dict
