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
        lambda_rc: float = 0.5,
        lambda_hu: float = 0.3,
        lambda_edge: float = 1.2,        # Strong edge preservation
        lambda_texture: float = 1.0,     # Strong texture preservation
        lambda_hf_noise: float = 3.0,    # Strong noise removal
        lambda_mid_noise: float = 2.0,   # Strong mid-freq removal
        lambda_syn: float = 0.5,
        lambda_ic: float = 0.1,
        min_body_pixels: int = 512,
        artifact_grad_factor: float = 1.9,
        flat_threshold: float = 0.09,
    ):
        super().__init__()
        self.lambda_rc = lambda_rc
        self.lambda_hu = lambda_hu
        self.lambda_edge = lambda_edge
        self.lambda_texture = lambda_texture
        self.lambda_hf_noise = lambda_hf_noise
        self.lambda_mid_noise = lambda_mid_noise
        self.lambda_syn = lambda_syn
        self.lambda_ic = lambda_ic
        
        self.min_body_pixels = min_body_pixels
        self.artifact_grad_factor = artifact_grad_factor
        self.flat_threshold = flat_threshold

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
        
        # Strong edge mask: includes vessels
        edge_mask = (grad_mag_input > 0.05)
        if edge_mask.sum() > 100:
            loss_edge = F.l1_loss(grad_mag_pred[edge_mask], grad_mag_input[edge_mask])
        else:
            loss_edge = F.l1_loss(grad_x_pred, grad_x_input) + F.l1_loss(grad_y_pred, grad_y_input)
        
        # ===== 4. STRONG TEXTURE PRESERVATION (fine vessels, trabecular bone) =====
        texture_mask = (grad_mag_input > self.flat_threshold) & (grad_mag_input < 0.5)
        
        if texture_mask.sum() > 100:
            # Preserve both magnitude AND direction
            loss_texture = (
                F.l1_loss(grad_mag_pred[texture_mask], grad_mag_input[texture_mask]) +
                0.5 * F.l1_loss(grad_x_pred[texture_mask], grad_x_input[texture_mask]) +
                0.5 * F.l1_loss(grad_y_pred[texture_mask], grad_y_input[texture_mask])
            )
        else:
            loss_texture = torch.tensor(0.0, device=y_pred.device)

        # ===== 5. MID-FREQ NOISE REMOVAL =====
        flat_mask = (grad_mag_input < self.flat_threshold)
        flat_body_mask = flat_mask & body_mask

        if flat_body_mask.sum() > 100:
            mid_sigma = 1.0
            low_sigma = 3.0
            
            low_input = self._gaussian_blur(x_i, low_sigma)
            low_pred = self._gaussian_blur(y_pred, low_sigma)
            
            mid_input = self._gaussian_blur(x_i, mid_sigma) - low_input
            mid_pred = self._gaussian_blur(y_pred, mid_sigma) - low_pred

            mid_input_abs = torch.abs(mid_input[flat_body_mask])
            mid_pred_abs = torch.abs(mid_pred[flat_body_mask])
            target_mid = 0.2 * mid_input_abs  # Aggressive: 20%
            loss_mid_noise = F.l1_loss(mid_pred_abs, target_mid)
        else:
            loss_mid_noise = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 6. HIGH-FREQ NOISE REMOVAL =====
        lap_input = self._laplacian(x_i)
        lap_pred = self._laplacian(y_pred)

        if flat_mask.sum() > 100:
            lap_input_abs = torch.abs(lap_input[flat_mask])
            lap_pred_abs = torch.abs(lap_pred[flat_mask])
            
            target_hf = 0.2 * lap_input_abs  # Aggressive: 20%
            loss_hf_noise = F.l1_loss(lap_pred_abs, target_hf)
        else:
            loss_hf_noise = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 7. SYNTHETIC NOISE SUPERVISION =====
        if flat_body_mask.sum() > 100:
            loss_syn = F.l1_loss(noise_pred[flat_body_mask], noise_synthetic[flat_body_mask])
        else:
            loss_syn = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 8. ARTIFACT SUPPRESSION (prevent new artifacts) =====
        if flat_mask.sum() > 100:
            artifact_penalty = torch.clamp(
                grad_mag_pred[flat_mask] - self.artifact_grad_factor * grad_mag_input[flat_mask],
                min=0
            ).mean()
            loss_ic = artifact_penalty
        else:
            loss_ic = torch.tensor(0.0, device=y_pred.device)
        
        # ===== TOTAL LOSS =====
        total_loss = (
            self.lambda_rc * loss_rc
            + self.lambda_hu * loss_hu
            + self.lambda_edge * loss_edge
            + self.lambda_texture * loss_texture
            + self.lambda_hf_noise * loss_hf_noise
            + self.lambda_mid_noise * loss_mid_noise
            + self.lambda_syn * loss_syn
            + self.lambda_ic * loss_ic
        )

        loss_dict = {
            "total": total_loss.item(),
            "rc": loss_rc.item(),
            "hu": loss_hu.item() if isinstance(loss_hu, torch.Tensor) else 0.0,
            "edge": loss_edge.item(),
            "texture": loss_texture.item() if isinstance(loss_texture, torch.Tensor) else 0.0,
            "hf_noise": loss_hf_noise.item() if isinstance(loss_hf_noise, torch.Tensor) else 0.0,
            "mid_noise": loss_mid_noise.item() if isinstance(loss_mid_noise, torch.Tensor) else 0.0,
            "syn": loss_syn.item() if isinstance(loss_syn, torch.Tensor) else 0.0,
            "ic": loss_ic.item() if isinstance(loss_ic, torch.Tensor) else 0.0,
        }
        
        return total_loss, loss_dict
    
    def _sobel_x(self, x):
        kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _sobel_y(self, x):
        kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _laplacian(self, x):
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)

    def _gaussian_blur(self, x, sigma: float):
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k += 1
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, k, k)
        return F.conv2d(x, kernel_2d, padding=k // 2)


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
        lambda_rc: float = 0.5,
        lambda_hu: float = 0.3,
        lambda_edge: float = 1.2,
        lambda_texture: float = 1.0,
        lambda_h_streak: float = 3.0,    # Horizontal streak (NPS-guided)
        lambda_v_streak: float = 1.1,    # Vertical streak (weaker)
        lambda_lf_artifact: float = 2.0, # Low-freq shading
        lambda_ic: float = 0.2,
        min_body_pixels: int = 512,
        artifact_grad_factor: float = 1.9,
        flat_threshold: float = 0.09,
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
        
        # Directional filters for streak detection
        # Horizontal streak: average along width (detect row-wise patterns)
        kernel_size = 15  # Larger for streak detection
        self.register_buffer('h_streak_kernel', 
                           torch.ones(1, 1, 1, kernel_size) / kernel_size)
        # Vertical streak: average along height
        self.register_buffer('v_streak_kernel', 
                           torch.ones(1, 1, kernel_size, 1) / kernel_size)

    def forward(self, y_pred, noise_pred, batch_dict):
        y_pred = y_pred.squeeze(2)
        noise_pred = noise_pred.squeeze(2)
        
        x_i = batch_dict["x_i"]
        x_mid = batch_dict["x_mid"]
        W = batch_dict["W"]
        
        # ===== 1. WEAK RECONSTRUCTION =====
        loss_rc = F.l1_loss(y_pred * W, x_i * W)
        
        # ===== 2. WEAK HU PRESERVATION =====
        body_mask = (x_i > 0.15) & (x_i < 0.85)
        if body_mask.sum() > self.min_body_pixels:
            mean_input = x_i[body_mask].mean()
            mean_pred = y_pred[body_mask].mean()
            loss_hu = F.l1_loss(mean_pred, mean_input)
        else:
            loss_hu = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 3. STRONG EDGE PRESERVATION =====
        grad_x_pred = self._sobel_x(y_pred)
        grad_y_pred = self._sobel_y(y_pred)
        grad_x_input = self._sobel_x(x_i)
        grad_y_input = self._sobel_y(x_i)
        
        grad_mag_input = torch.sqrt(grad_x_input**2 + grad_y_input**2 + 1e-8)
        grad_mag_pred = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-8)
        
        edge_mask = (grad_mag_input > 0.05)
        if edge_mask.sum() > 100:
            loss_edge = F.l1_loss(grad_mag_pred[edge_mask], grad_mag_input[edge_mask])
        else:
            loss_edge = F.l1_loss(grad_x_pred, grad_x_input) + F.l1_loss(grad_y_pred, grad_y_input)
        
        # ===== 4. STRONG TEXTURE PRESERVATION =====
        texture_mask = (grad_mag_input > self.flat_threshold) & (grad_mag_input < 0.5)
        
        if texture_mask.sum() > 100:
            loss_texture = (
                F.l1_loss(grad_mag_pred[texture_mask], grad_mag_input[texture_mask]) +
                0.5 * F.l1_loss(grad_x_pred[texture_mask], grad_x_input[texture_mask]) +
                0.5 * F.l1_loss(grad_y_pred[texture_mask], grad_y_input[texture_mask])
            )
        else:
            loss_texture = torch.tensor(0.0, device=y_pred.device)

        # ===== 5. HORIZONTAL STREAK REMOVAL (NPS-guided) =====
        flat_mask = (grad_mag_input < self.flat_threshold)
        flat_body_mask = flat_mask & body_mask
        
        if flat_body_mask.sum() > 100:
            # Detect row-wise patterns by averaging along width
            h_resp_input = self._detect_horizontal_streak(x_i)
            h_resp_pred = self._detect_horizontal_streak(y_pred)
            
            h_input_abs = torch.abs(h_resp_input[flat_body_mask])
            h_pred_abs = torch.abs(h_resp_pred[flat_body_mask])
            
            # Aggressive: reduce to 15%
            target_h = 0.15 * h_input_abs
            loss_h_streak = F.l1_loss(h_pred_abs, target_h)
        else:
            loss_h_streak = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 6. VERTICAL STREAK REMOVAL =====
        if flat_body_mask.sum() > 100:
            v_resp_input = self._detect_vertical_streak(x_i)
            v_resp_pred = self._detect_vertical_streak(y_pred)
            
            v_input_abs = torch.abs(v_resp_input[flat_body_mask])
            v_pred_abs = torch.abs(v_resp_pred[flat_body_mask])
            
            target_v = 0.15 * v_input_abs
            loss_v_streak = F.l1_loss(v_pred_abs, target_v)
        else:
            loss_v_streak = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 7. LOW-FREQ SHADING REMOVAL =====
        if flat_body_mask.sum() > 100:
            low_sigma = 5.0
            low_input = self._gaussian_blur(x_i, low_sigma)
            low_pred = self._gaussian_blur(y_pred, low_sigma)
            low_mid = self._gaussian_blur(x_mid, low_sigma)

            lf_input = (low_input - low_mid)[flat_body_mask]
            lf_pred = (low_pred - low_mid)[flat_body_mask]
            
            # Aggressive: reduce to 20%
            loss_lf_artifact = F.l1_loss(lf_pred, lf_input * 0.2)
        else:
            loss_lf_artifact = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 8. ARTIFACT SUPPRESSION =====
        if flat_mask.sum() > 100:
            artifact_penalty = torch.clamp(
                grad_mag_pred[flat_mask] - self.artifact_grad_factor * grad_mag_input[flat_mask],
                min=0
            ).mean()
            loss_ic = artifact_penalty
        else:
            loss_ic = torch.tensor(0.0, device=y_pred.device)
        
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

        loss_dict = {
            "total": total_loss.item(),
            "rc": loss_rc.item(),
            "hu": loss_hu.item() if isinstance(loss_hu, torch.Tensor) else 0.0,
            "edge": loss_edge.item(),
            "texture": loss_texture.item() if isinstance(loss_texture, torch.Tensor) else 0.0,
            "h_streak": loss_h_streak.item() if isinstance(loss_h_streak, torch.Tensor) else 0.0,
            "v_streak": loss_v_streak.item() if isinstance(loss_v_streak, torch.Tensor) else 0.0,
            "lf_artifact": loss_lf_artifact.item() if isinstance(loss_lf_artifact, torch.Tensor) else 0.0,
            "ic": loss_ic.item() if isinstance(loss_ic, torch.Tensor) else 0.0,
        }
        
        return total_loss, loss_dict
    
    def _detect_horizontal_streak(self, x):
        """Detect horizontal streaks by averaging along width"""
        x_pad = F.pad(x, (self.h_streak_kernel.size(3)//2, self.h_streak_kernel.size(3)//2, 0, 0), mode='replicate')
        h_avg = F.conv2d(x_pad, self.h_streak_kernel)
        # High-pass: original - smoothed = streaks
        h_response = x - h_avg
        return h_response
    
    def _detect_vertical_streak(self, x):
        """Detect vertical streaks by averaging along height"""
        x_pad = F.pad(x, (0, 0, self.v_streak_kernel.size(2)//2, self.v_streak_kernel.size(2)//2), mode='replicate')
        v_avg = F.conv2d(x_pad, self.v_streak_kernel)
        v_response = x - v_avg
        return v_response
    
    def _sobel_x(self, x):
        kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _sobel_y(self, x):
        kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)

    def _gaussian_blur(self, x, sigma: float):
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k += 1
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
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