"""
Loss functions for Residual-based NS-N2N CT Denoising

Plan B: Input-based ROI to prevent "scale-down" cheating
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighQualityNSN2NLoss(nn.Module):
    """
    Plan B Loss: Structure/HU preservation FIRST, noise reduction LATER
    
    Key changes from original:
    1. Noise loss uses INPUT-based ROI (not output-based)
    2. Flat region detection prevents structure/edge confusion
    3. Mean-centered std measurement blocks HU scale manipulation
    """
    
    def __init__(
        self,
        lambda_rc: float = 2.0,
        lambda_noise: float = 0.05,
        lambda_edge: float = 0.7,
        lambda_hf: float = 0.05,
        lambda_hu: float = 1.0,
        lambda_ic: float = 0.1,
        target_noise_ratio: float = 0.7,
        min_body_pixels: int = 512,
        artifact_grad_factor: float = 1.9,
        flat_threshold: float = 0.15,
    ):
        super().__init__()
        self.lambda_rc = lambda_rc
        self.lambda_noise = lambda_noise
        self.lambda_edge = lambda_edge
        self.lambda_hf = lambda_hf
        self.lambda_hu = lambda_hu
        self.lambda_ic = lambda_ic
        
        self.target_noise_ratio = target_noise_ratio
        self.min_body_pixels = min_body_pixels
        self.artifact_grad_factor = artifact_grad_factor
        self.flat_threshold = flat_threshold
    
    def forward(self, y_pred, batch_dict):
        """
        Args:
            y_pred: (B, 1, 1, H, W) - denoised output from model
            batch_dict: Dictionary containing x_i, x_ip1, x_mid, W, noise_synthetic
        
        Returns:
            total_loss, loss_dict
        """
        # Squeeze depth dimension from y_pred: (B, 1, 1, H, W) -> (B, 1, H, W)
        y_pred = y_pred.squeeze(2)
        
        x_i = batch_dict["x_i"]
        x_ip1 = batch_dict["x_ip1"]
        x_mid = batch_dict["x_mid"]
        W = batch_dict["W"]
        
        # ===== 1. RECONSTRUCTION LOSS (Structure preservation) =====
        loss_rc = F.l1_loss(y_pred * W, x_mid * W)
        
        # ===== 2. NOISE CONTROL LOSS (Plan B: Input-based flat ROI) =====
        # Use INPUT-based tissue mask (not output-based!)
        tissue_mask = (x_i > 0.2) & (x_i < 0.8)
        
        # Detect flat regions in INPUT using gradients
        grad_x_input = self._sobel_x(x_i)
        grad_y_input = self._sobel_y(x_i)
        grad_mag_input = torch.sqrt(grad_x_input**2 + grad_y_input**2 + 1e-8)
        flat_mask = (grad_mag_input < self.flat_threshold)
        
        # Noise ROI = body AND flat (avoid edges/structure)
        noise_roi = tissue_mask & flat_mask
        
        if noise_roi.sum() > self.min_body_pixels:
            # Mean-centered std measurement (blocks HU scale manipulation)
            yp_roi = y_pred[noise_roi]
            xi_roi = x_i[noise_roi]
            
            denoised_std = (yp_roi - yp_roi.mean()).std()
            input_std = (xi_roi - xi_roi.mean()).std()
            target_std = input_std * self.target_noise_ratio
            
            loss_noise = F.l1_loss(denoised_std, target_std)
        else:
            loss_noise = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 3. EDGE PRESERVATION LOSS =====
        grad_x_pred = self._sobel_x(y_pred)
        grad_y_pred = self._sobel_y(y_pred)
        
        loss_edge = F.l1_loss(grad_x_pred, grad_x_input) + F.l1_loss(grad_y_pred, grad_y_input)
        
        # ===== 4. HU PRESERVATION LOSS =====
        body_mask = (x_i > 0.15) & (x_i < 0.85)
        if body_mask.sum() > self.min_body_pixels:
            mean_input = x_i[body_mask].mean()
            mean_pred = y_pred[body_mask].mean()
            loss_hu = F.l1_loss(mean_pred, mean_input)
        else:
            loss_hu = torch.tensor(0.0, device=y_pred.device)
        
        # ===== 5. HIGH-FREQUENCY LOSS =====
        laplacian_pred = self._laplacian(y_pred)
        laplacian_input = self._laplacian(x_i)
        loss_hf = F.l1_loss(laplacian_pred, laplacian_input)
        
        # ===== 6. ARTIFACT SUPPRESSION LOSS =====
        grad_mag_pred = torch.sqrt(grad_x_pred**2 + grad_y_pred**2 + 1e-8)
        
        if flat_mask.sum() > 100:
            # Penalize strong gradients in flat regions
            artifact_penalty = torch.clamp(
                grad_mag_pred[flat_mask] - self.artifact_grad_factor * grad_mag_input[flat_mask],
                min=0
            ).mean()
            loss_ic = artifact_penalty
        else:
            loss_ic = torch.tensor(0.0, device=y_pred.device)
        
        # ===== TOTAL LOSS =====
        total_loss = (
            self.lambda_rc * loss_rc +
            self.lambda_noise * loss_noise +
            self.lambda_edge * loss_edge +
            self.lambda_hf * loss_hf +
            self.lambda_hu * loss_hu +
            self.lambda_ic * loss_ic
        )
        
        loss_dict = {
            "total": total_loss.item(),
            "rc": loss_rc.item(),
            "noise": loss_noise.item() if isinstance(loss_noise, torch.Tensor) else 0.0,
            "edge": loss_edge.item(),
            "hf": loss_hf.item(),
            "hu": loss_hu.item() if isinstance(loss_hu, torch.Tensor) else 0.0,
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