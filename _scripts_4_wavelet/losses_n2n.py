"""
Loss functions for Residual-based NS-N2N CT Denoising

FOCUS: Remove TRUE noise (high-freq grain), NOT just darken images
Key principle: Target high-frequency noise in flat regions only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighQualityNSN2NLoss(nn.Module):
    def __init__(
        self,
        lambda_rc: float = 2.0,
        lambda_hu: float = 1.5,
        lambda_edge: float = 1.0,
        lambda_texture: float = 0.8,
        lambda_hf_noise: float = 1.5,
        lambda_syn: float = 1.0,
        lambda_ic: float = 0.8,
        # ★ 추가: 중주파 / 저주파 가중치
        lambda_mid_noise: float = 0.8,
        lambda_lf_artifact: float = 0.6,
        min_body_pixels: int = 4096,
        artifact_grad_factor: float = 2.0,
        flat_threshold: float = 0.03,
    ):
        super().__init__()
        self.lambda_rc = lambda_rc
        self.lambda_hu = lambda_hu
        self.lambda_edge = lambda_edge
        self.lambda_texture = lambda_texture
        self.lambda_hf_noise = lambda_hf_noise
        self.lambda_syn = lambda_syn
        self.lambda_ic = lambda_ic
        # ★ 추가
        self.lambda_mid_noise = lambda_mid_noise
        self.lambda_lf_artifact = lambda_lf_artifact

        self.min_body_pixels = min_body_pixels
        self.artifact_grad_factor = artifact_grad_factor
        self.flat_threshold = flat_threshold

    
    def forward(self, y_pred, noise_pred, batch_dict):
        """
        Args:
            y_pred: (B, 1, 1, H, W) - denoised output
            noise_pred: (B, 1, 1, H, W) - predicted noise
            batch_dict: Dictionary containing x_i, x_ip1, x_mid, W, noise_synthetic
        
        Returns:
            total_loss, loss_dict
        """
        # Squeeze depth dimension
        y_pred = y_pred.squeeze(2)
        noise_pred = noise_pred.squeeze(2)
        
        x_i = batch_dict["x_i"]
        x_ip1 = batch_dict["x_ip1"]
        x_mid = batch_dict["x_mid"]
        W = batch_dict["W"]
        
        # ===== 1. RECONSTRUCTION LOSS (Supervised: target = clean origin) =====
        loss_rc = F.l1_loss(y_pred * W, x_i * W)
        
        # ===== 2. STRONG HU PRESERVATION (Prevent darkening!) =====
        body_mask = (x_i > 0.15) & (x_i < 0.85)
        if body_mask.sum() > self.min_body_pixels:
            # Mean HU must match exactly (MSE for stronger penalty)
            mean_input = x_i[body_mask].mean()
            mean_pred = y_pred[body_mask].mean()
            loss_hu = F.mse_loss(mean_pred, mean_input)
            
            # Percentiles should also match (prevent contrast shift/darkening)
            input_25 = torch.quantile(x_i[body_mask], 0.25)
            input_75 = torch.quantile(x_i[body_mask], 0.75)
            pred_25 = torch.quantile(y_pred[body_mask], 0.25)
            pred_75 = torch.quantile(y_pred[body_mask], 0.75)
            loss_hu = loss_hu + 0.5 * (F.mse_loss(pred_25, input_25) + F.mse_loss(pred_75, input_75))
        else:
            loss_hu = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 3. EDGE PRESERVATION LOSS =====
        grad_x_pred = self._sobel_x(y_pred)
        grad_y_pred = self._sobel_y(y_pred)
        grad_x_input = self._sobel_x(x_i)
        grad_y_input = self._sobel_y(x_i)
        
        loss_edge = F.l1_loss(grad_x_pred, grad_x_input) + F.l1_loss(grad_y_pred, grad_y_input)
        
        # ===== 4. TEXTURE PRESERVATION (Fine anatomical structure) =====
        # Identify textured regions (not flat, not strong edges)
        grad_mag_input = torch.sqrt(grad_x_input**2 + grad_y_input**2 + 1e-8)
        grad_mag_pred = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-8)
        
        # Texture mask: moderate gradients
        texture_mask = (grad_mag_input > self.flat_threshold) & (grad_mag_input < 0.5)
        
        if texture_mask.sum() > 100:
            # Preserve gradient magnitude in textured regions (prevents blur)
            loss_texture = F.l1_loss(grad_mag_pred[texture_mask], grad_mag_input[texture_mask])
        else:
            loss_texture = torch.tensor(0.0, device=y_pred.device)

        # ===== 4.5 MULTI-BAND NOISE / ARTIFACT REDUCTION =====
        # flat + body 영역에서 저주파 shading, 중주파 banding 까지 제어
        flat_mask = (grad_mag_input < self.flat_threshold)
        flat_body_mask = flat_mask & body_mask

        if flat_body_mask.sum() > 100:
            # --- Low-frequency (shading) component ---
            low_sigma = 3.0
            low_input = self._gaussian_blur(x_i, low_sigma)
            low_pred  = self._gaussian_blur(y_pred, low_sigma)
            low_mid   = self._gaussian_blur(x_mid, low_sigma)  # neighbor 평균

            # 이웃(slice 평균)과의 저주파 차이가 줄어들도록 (banding/shading 완화)
            lf_input = (low_input - low_mid)[flat_body_mask]
            lf_pred  = (low_pred  - low_mid)[flat_body_mask]
            loss_lf_artifact = F.l1_loss(lf_pred, lf_input * 0.3)  # 30%로 줄이기

            # --- Mid-frequency (banding) component ---
            mid_sigma = 1.0
            mid_input = self._gaussian_blur(x_i, mid_sigma) - low_input
            mid_pred  = self._gaussian_blur(y_pred, mid_sigma) - low_pred

            mid_input_abs = torch.abs(mid_input[flat_body_mask])
            mid_pred_abs  = torch.abs(mid_pred[flat_body_mask])
            target_mid = 0.3 * mid_input_abs  # mid-band 도 30% 수준으로
            loss_mid_noise = F.l1_loss(mid_pred_abs, target_mid)
        else:
            loss_lf_artifact = torch.tensor(0.0, device=y_pred.device)
            loss_mid_noise   = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 5. HIGH-FREQ NOISE REMOVAL (Target ACTUAL noise) =====
        # Use Laplacian (2nd derivative) to detect high-frequency components
        lap_input = self._laplacian(x_i)
        lap_pred = self._laplacian(y_pred)

        if flat_mask.sum() > 100:
            # In flat regions: actively reduce high-freq (noise)
            lap_input_abs = torch.abs(lap_input[flat_mask])
            lap_pred_abs = torch.abs(lap_pred[flat_mask])
            
            # Target: output HF should be 30% of input HF
            # This actively removes noise while keeping some texture
            target_hf = 0.3 * lap_input_abs
            loss_hf_noise = F.l1_loss(lap_pred_abs, target_hf)
        else:
            loss_hf_noise = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 6. SYNTHETIC NOISE SUPERVISION =====
        noise_synthetic = batch_dict["noise_synthetic"]
        flat_body_mask = flat_mask & body_mask
        
        if flat_body_mask.sum() > 100:
            # Direct supervision: predicted noise should match synthetic noise
            loss_syn = F.l1_loss(noise_pred[flat_body_mask], noise_synthetic[flat_body_mask])
        else:
            loss_syn = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 7. ARTIFACT SUPPRESSION LOSS =====
        if flat_mask.sum() > 100:
            # Penalize strong gradients where input is flat
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
            + self.lambda_lf_artifact * loss_lf_artifact
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
            "lf_artifact": loss_lf_artifact.item() if isinstance(loss_lf_artifact, torch.Tensor) else 0.0,
            "syn": loss_syn.item() if isinstance(loss_syn, torch.Tensor) else 0.0,
            "ic": loss_ic.item() if isinstance(loss_ic, torch.Tensor) else 0.0,
        }
        
        return total_loss, loss_dict
    
    def _sobel_x(self, x):
        """Sobel operator for horizontal gradients"""
        kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _sobel_y(self, x):
        """Sobel operator for vertical gradients"""
        kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)
    
    def _laplacian(self, x):
        """Laplacian operator for high-frequency detection"""
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x, kernel, padding=1)

    def _gaussian_blur(self, x, sigma: float):
        """
        Simple 2D Gaussian blur using depthwise conv2d.
        x: (B,1,H,W)
        """
        # kernel size ≈ 6*sigma+1
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k += 1
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        gauss_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
        kernel_2d = kernel_2d.view(1, 1, k, k)
        return F.conv2d(x, kernel_2d, padding=k // 2)
