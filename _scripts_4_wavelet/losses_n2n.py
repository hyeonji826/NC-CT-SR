"""
Loss functions for Two-Stage CT Denoising
Stage 1: Noise Removal (Random high-freq grain)
Stage 2: Artifact Removal (Directional streaks, shading)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# STAGE 1: NOISE REMOVAL LOSS
# Focus: Remove random high-frequency noise while preserving structure
# ============================================================================

class NoiseRemovalLoss(nn.Module):
    """
    Stage 1: Random Noise Removal
    - Target: Remove high/mid frequency random noise (grain)
    - Preserve: All anatomical structures, edges, fine vessels
    - Avoid: Darkening, smoothing, contrast shift
    """
    def __init__(
        self,
        lambda_rc: float,
        lambda_hu: float,
        lambda_edge: float,
        lambda_hf_edge: float,
        lambda_hf_flat: float,
        lambda_syn: float,
        lambda_ic: float,
        min_body_pixels: int,
        artifact_grad_factor: float,
        flat_threshold: float,
        hf_target_ratio: float,
        edge_threshold: float = 0.05,  # Edge detection threshold
    ):
        super().__init__()
        self.lambda_rc = lambda_rc
        self.lambda_hu = lambda_hu
        self.lambda_edge = lambda_edge
        self.lambda_hf_edge = lambda_hf_edge
        self.lambda_hf_flat = lambda_hf_flat
        self.lambda_syn = lambda_syn
        self.lambda_ic = lambda_ic
        
        self.min_body_pixels = min_body_pixels
        self.artifact_grad_factor = artifact_grad_factor
        self.flat_threshold = flat_threshold
        self.edge_threshold = edge_threshold
        self.hf_target_ratio = hf_target_ratio

    def forward(self, y_pred, noise_pred, batch_dict):
        y_pred = y_pred.squeeze(2)
        noise_pred = noise_pred.squeeze(2)
        
        x_i = batch_dict["x_i"]
        W = batch_dict["W"]
        noise_synthetic = batch_dict["noise_synthetic"]
        
        # ===== 1. WEAK RECONSTRUCTION (allow deviation from noisy origin) =====
        loss_rc = F.l1_loss(y_pred * W, x_i * W)
        
        # ===== 2. WEAK HU PRESERVATION (prevent darkening only) =====
        body_mask = (x_i > 0.15) & (x_i < 0.85)
        if body_mask.sum() > self.min_body_pixels:
            mean_input = x_i[body_mask].mean()
            mean_pred = y_pred[body_mask].mean()
            loss_hu = F.l1_loss(mean_pred, mean_input)  # L1 instead of MSE (weaker)
        else:
            loss_hu = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 3. STRONG EDGE PRESERVATION =====
        grad_x_pred = self._sobel_x(y_pred)
        grad_y_pred = self._sobel_y(y_pred)
        grad_x_input = self._sobel_x(x_i)
        grad_y_input = self._sobel_y(x_i)
        
        grad_mag_input = torch.sqrt(grad_x_input**2 + grad_y_input**2 + 1e-8)
        grad_mag_pred = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-8)
        
        edge_mask = (grad_mag_input > self.edge_threshold)
        
        if edge_mask.sum() > 100:
            loss_edge = F.l1_loss(
                grad_mag_pred[edge_mask],
                grad_mag_input[edge_mask],
            )
        else:
            loss_edge = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 4. HF EDGE STRUCTURE PRESERVATION (L_hf_edge) =====
        lap_input = self._laplacian(x_i)
        lap_pred = self._laplacian(y_pred)
        
        edge_mask = (grad_mag_input > self.edge_threshold)
        
        if edge_mask.sum() > 100:
            # Edge 영역에서 HF를 origin과 동일하게 유지
            loss_hf_edge = F.l1_loss(lap_pred[edge_mask], lap_input[edge_mask])
        else:
            loss_hf_edge = torch.tensor(0.0, device=y_pred.device)

        # ===== 5. FLAT 영역 HF NOISE 제거 (L_hf_flat) =====
        flat_mask = (grad_mag_input < self.flat_threshold)
        flat_body_mask = flat_mask & body_mask
        
        if flat_body_mask.sum() > 100:
            lap_input_abs = torch.abs(lap_input[flat_body_mask])
            lap_pred_abs = torch.abs(lap_pred[flat_body_mask])
            # Flat 영역에서 HF를 target ratio로 감소 (덜 공격적)
            target_hf = self.hf_target_ratio * lap_input_abs
            loss_hf_flat = F.l1_loss(lap_pred_abs, target_hf)
        else:
            loss_hf_flat = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 6. SYNTHETIC NOISE SUPERVISION =====
        if flat_body_mask.sum() > 100:
            loss_syn = F.l1_loss(noise_pred[flat_body_mask], noise_synthetic[flat_body_mask])
        else:
            loss_syn = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 7. ARTIFACT SUPPRESSION (prevent new artifacts) =====
        grad_mag_ratio = grad_mag_pred / (grad_mag_input + 1e-8)
        artifact_penalty = torch.clamp(
            grad_mag_ratio - self.artifact_grad_factor,
            min=0.0
        )
        loss_ic = artifact_penalty.mean()
        
        # ===== TOTAL LOSS =====
        total_loss = (
            self.lambda_rc * loss_rc
            + self.lambda_hu * loss_hu
            + self.lambda_edge * loss_edge
            + self.lambda_hf_edge * loss_hf_edge
            + self.lambda_hf_flat * loss_hf_flat
            + self.lambda_syn * loss_syn
            + self.lambda_ic * loss_ic
        )
        
        return total_loss, {
            "rc": loss_rc.detach(),
            "hu": loss_hu.detach(),
            "edge": loss_edge.detach(),
            "hf_edge": loss_hf_edge.detach(),
            "hf_flat": loss_hf_flat.detach(),
            "syn": loss_syn.detach(),
            "ic": loss_ic.detach(),
        }
    
    # ----------------------------------------------------------------------
    # Helper filters
    # ----------------------------------------------------------------------
    def _sobel_x(self, x):
        """Sobel filter in x-direction"""
        kernel = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _sobel_y(self, x):
        """Sobel filter in y-direction"""
        kernel = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _laplacian(self, x):
        """Laplacian filter for HF component"""
        kernel = torch.tensor(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)


# ============================================================================
# STAGE 2: ARTIFACT REMOVAL LOSS
# Focus: Remove directional streaks and low-freq shading
# ============================================================================

class ArtifactRemovalLoss(nn.Module):
    """
    Stage 2: Directional Artifact Removal
    - Target: Horizontal streaks, vertical streaks, low-freq shading
    - Preserve: All structures (already denoised from Stage 1)
    - Based on: NPS analysis showing H:V = 2.7:1 anisotropy
    """
    def __init__(
        self,
        lambda_rc: float,
        lambda_hu: float,
        lambda_edge: float,
        lambda_texture: float,
        lambda_h_streak: float,
        lambda_v_streak: float,
        lambda_lf_artifact: float,
        lambda_ic: float,
        min_body_pixels: int,
        artifact_grad_factor: float,
        flat_threshold: float,
        edge_threshold: float = 0.05,
    ):
        super().__init__()
        self.lambda_rc = lambda_rc
        self.lambda_hu = lambda_hu
        self.lambda_edge = lambda_edge
        self.lambda_texture = lambda_texture
        self.lambda_h_streak = lambda_h_streak
        self.lambda_v_streak = lambda_v_streak
        self.lambda_lf_artifact = lambda_lf_artifact
        self.lambda_ic = lambda_ic
        
        self.min_body_pixels = min_body_pixels
        self.artifact_grad_factor = artifact_grad_factor
        self.flat_threshold = flat_threshold
        self.edge_threshold = edge_threshold
        
        # Directional filters for streak detection
        # Horizontal streak: 1D kernel along x (W)
        self.register_buffer(
            "h_kernel",
            torch.ones(1, 1, 1, 15) / 15.0,
        )
        # Vertical streak: 1D kernel along y (H)
        self.register_buffer(
            "v_kernel",
            torch.ones(1, 1, 15, 1) / 15.0,
        )
    
    def forward(self, y_pred, noise_pred, batch_dict):
        # noise_pred는 Stage 2에서는 사용하지 않음 (인터페이스 통일용)
        _ = noise_pred
        
        y_pred = y_pred.squeeze(2)
        x_i = batch_dict["x_i"]
        W = batch_dict["W"]
        
        # ===== 1. RECONSTRUCTION (L1) =====
        loss_rc = F.l1_loss(y_pred * W, x_i * W)
        
        # ===== 2. HU PRESERVATION =====
        body_mask = (x_i > 0.15) & (x_i < 0.85)
        if body_mask.sum() > self.min_body_pixels:
            mean_input = x_i[body_mask].mean()
            mean_pred = y_pred[body_mask].mean()
            loss_hu = F.l1_loss(mean_pred, mean_input)
        else:
            loss_hu = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 3. EDGE PRESERVATION =====
        grad_x_pred = self._sobel_x(y_pred)
        grad_y_pred = self._sobel_y(y_pred)
        grad_x_input = self._sobel_x(x_i)
        grad_y_input = self._sobel_y(x_i)
        
        grad_mag_input = torch.sqrt(grad_x_input**2 + grad_y_input**2 + 1e-8)
        grad_mag_pred = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-8)
        
        edge_mask = (grad_mag_input > self.edge_threshold)
        if edge_mask.sum() > 100:
            loss_edge = F.l1_loss(
                grad_mag_pred[edge_mask],
                grad_mag_input[edge_mask],
            )
        else:
            loss_edge = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 4. TEXTURE PRESERVATION (L_texture) =====
        # Body 영역에서 gradient 구조를 유지
        if body_mask.sum() > self.min_body_pixels:
            loss_texture = (
                F.l1_loss(grad_x_pred[body_mask], grad_x_input[body_mask])
                + F.l1_loss(grad_y_pred[body_mask], grad_y_input[body_mask])
            ) * 0.5
        else:
            loss_texture = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 5. DIRECTIONAL STREAK SUPPRESSION =====
        # Horizontal (x 방향으로 긴 줄무늬)
        h_input = F.conv2d(x_i, self.h_kernel, padding=(0, 7))
        h_pred = F.conv2d(y_pred, self.h_kernel, padding=(0, 7))
        # Vertical
        v_input = F.conv2d(x_i, self.v_kernel, padding=(7, 0))
        v_pred = F.conv2d(y_pred, self.v_kernel, padding=(7, 0))
        
        # Body mask를 conv 크기에 맞춰 crop
        body_mask_h = body_mask[:, :, :, 7:-7] if h_input.shape[-1] == body_mask.shape[-1] - 14 else body_mask
        body_mask_v = body_mask[:, :, 7:-7, :] if v_input.shape[-2] == body_mask.shape[-2] - 14 else body_mask
        
        if body_mask_h.sum() > self.min_body_pixels:
            # Horizontal streak는 더 강하게 억제
            target_h = 0.15 * torch.abs(h_input[body_mask_h])
            loss_h_streak = F.l1_loss(torch.abs(h_pred[body_mask_h]), target_h)
        else:
            loss_h_streak = torch.tensor(0.0, device=y_pred.device)
        
        if body_mask_v.sum() > self.min_body_pixels:
            target_v = 0.4 * torch.abs(v_input[body_mask_v])
            loss_v_streak = F.l1_loss(torch.abs(v_pred[body_mask_v]), target_v)
        else:
            loss_v_streak = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 6. LOW-FREQ ARTIFACT SUPPRESSION (L_lf_artifact) =====
        lf_input = self._gaussian_blur(x_i, sigma=25.0)
        lf_pred = self._gaussian_blur(y_pred, sigma=25.0)
        
        if body_mask.sum() > self.min_body_pixels:
            loss_lf_artifact = F.l1_loss(
                lf_pred[body_mask],
                lf_input[body_mask],
            )
        else:
            loss_lf_artifact = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 7. ARTIFACT CONSISTENCY (IC) =====
        grad_mag_ratio = grad_mag_pred / (grad_mag_input + 1e-8)
        artifact_penalty = torch.clamp(
            grad_mag_ratio - self.artifact_grad_factor,
            min=0.0,
        )
        loss_ic = artifact_penalty.mean()
        
        # ===== TOTAL LOSS =====
        total_loss = (
            self.lambda_rc * loss_rc
            + self.lambda_hu * loss_hu
            + self.lambda_edge * loss_edge
            + self.lambda_texture * loss_texture
            + self.lambda_h_streak * loss_h_streak
            + self.lambda_v_streak * loss_v_streak
            + self.lambda_lf_artifact * loss_lf_artifact
            + self.lambda_ic * loss_ic
        )
        
        return total_loss, {
            "rc": loss_rc.detach(),
            "hu": loss_hu.detach(),
            "edge": loss_edge.detach(),
            "texture": loss_texture.detach(),
            "h_streak": loss_h_streak.detach(),
            "v_streak": loss_v_streak.detach(),
            "lf_artifact": loss_lf_artifact.detach(),
            "ic": loss_ic.detach(),
        }
    
    # ----------------------------------------------------------------------
    # Helper filters
    # ----------------------------------------------------------------------
    def _sobel_x(self, x):
        kernel = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _sobel_y(self, x):
        kernel = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _gaussian_blur(self, x, sigma: float):
        """Approximate Gaussian blur via separable conv"""
        # kernel size ~ 6*sigma
        k = int(6 * sigma) | 1  # odd
        if k < 3:
            return x
        # 1D Gaussian
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        gauss_1d = torch.exp(-(coords**2) / (2 * sigma * sigma))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, k, k)
        return F.conv2d(x, kernel_2d, padding=k // 2)


# ============================================================================
# LEGACY: Original combined loss (for backward compatibility)
# ============================================================================

class HighQualityNSN2NLoss(nn.Module):
    """Original combined loss - use NoiseRemovalLoss or ArtifactRemovalLoss instead"""
    def __init__(self, **kwargs):
        super().__init__()
        # Default to Stage 1 behavior
        self.loss_fn = NoiseRemovalLoss(**kwargs)
    
    def forward(self, y_pred, noise_pred, batch_dict):
        return self.loss_fn(y_pred, noise_pred, batch_dict)