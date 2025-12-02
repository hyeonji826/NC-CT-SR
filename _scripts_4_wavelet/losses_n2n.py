# losses_n2n.py
# High-Quality NS-N2N Loss for Ultra-Low-Dose CT Enhancement
# Comprehensive loss combining structure preservation, noise control, and artifact suppression

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HighQualityNSN2NLoss(nn.Module):
    """
    High-quality CT denoising loss for NS-N2N framework
    
    Model output interpretation: y_i = F(x_i_aug) where y_i is the denoised/enhanced image
    
    Loss Components:
      1. L_recon : NS-N2N reconstruction - y_i ≈ x_{i+1} (neighbor consistency in matched regions W)
      2. L_rc    : Self-consistency - y_i ≈ x_i (content preservation in matched regions)
      3. L_edge  : Edge structure preservation - gradient similarity in NON-ARTIFACT regions only
      4. L_hf    : High-frequency regularization - prevents over-sharpening and residual noise
      5. L_noise : Noise level control - body region std(y_i) ≈ target_ratio * std(x_i)
      6. L_art   : Artifact suppression - selective smoothing of gradient outlier regions
    
    Key Design Decision:
      - L_art identifies artifact candidates (gradient outliers: mean + k*std threshold)
      - L_edge applies ONLY to non-artifact regions (1 - A)
      - This prevents conflict: edge preservation vs artifact removal on same pixels
      - Normal tissue/bone boundaries preserved, streak artifacts removed
    
    All parameters are configurable via config.yaml
    """

    def __init__(
        self,
        lambda_rc: float = 0.5,
        lambda_ic: float = 0.3,
        lambda_noise: float = 0.5,
        lambda_edge: float = 0.2,
        lambda_hf: float = 0.1,
        target_noise_ratio: float = 0.6,
        min_body_pixels: int = 512,
        artifact_grad_factor: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_rc = float(lambda_rc)
        self.lambda_ic = float(lambda_ic)
        self.lambda_noise = float(lambda_noise)
        self.lambda_edge = float(lambda_edge)
        self.lambda_hf = float(lambda_hf)

        self.target_noise_ratio = float(target_noise_ratio)
        self.min_body_pixels = int(min_body_pixels)
        self.artifact_grad_factor = float(artifact_grad_factor)

    def sobel_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Sobel gradient magnitude for edge detection"""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [ 0, 0, 0], [ 1, 2, 1]],
            dtype=x.dtype, device=x.device
        ).view(1, 1, 3, 3)

        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        return grad

    def low_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Simple low-pass filter using local mean (5x5 kernel)"""
        x_pad = F.pad(x, (2, 2, 2, 2), mode="reflect")
        lp = F.avg_pool2d(x_pad, kernel_size=5, stride=1)
        return lp

    def high_freq(self, x: torch.Tensor) -> torch.Tensor:
        """Extract high-frequency components"""
        lp = self.low_pass(x)
        return x - lp

    def forward(
        self,
        noise_pred: torch.Tensor,
        x_i: torch.Tensor,
        x_i_aug: torch.Tensor,
        x_ip1: torch.Tensor,
        x_mid: torch.Tensor,
        W: torch.Tensor,
        noise_synthetic: torch.Tensor,
    ):
        """
        Forward pass computing all loss components
        
        Args:
            noise_pred: Model output (denoised image y_i)
            x_i: Original noisy slice i
            x_i_aug: Augmented input (for training variance)
            x_ip1: Adjacent noisy slice i+1
            x_mid: Average of x_i and x_ip1 (compatibility, unused)
            W: Weight map (matched regions mask, 0-1)
            noise_synthetic: Synthetic noise added (compatibility, unused)
        
        Returns:
            total_loss: Weighted sum of all components
            log_dict: Individual loss values for monitoring
        """
        W = W.detach()
        y_i = noise_pred

        # 1) NS-N2N reconstruction: y_i ≈ x_{i+1} in matched regions
        diff_recon = (y_i - x_ip1) * W
        l_recon = diff_recon.pow(2).mean()

        # 2) Self-consistency: y_i ≈ x_i for content preservation
        diff_rc = (y_i - x_i) * W
        l_rc = diff_rc.pow(2).mean()

        # Prepare body mask for subsequent losses
        body_mask = (W > 0.5).float()

        # 6) Artifact suppression (compute before edge to get artifact mask)
        grad_x_full = self.sobel_gradient(x_i)
        mean_g = grad_x_full.mean(dim=[2, 3], keepdim=True)
        std_g = grad_x_full.std(dim=[2, 3], keepdim=True)
        thr = mean_g + self.artifact_grad_factor * std_g

        A = (grad_x_full > thr).float()
        A = A * body_mask
        non_artifact = (1.0 - A)

        if A.sum() > 0:
            lp_x = self.low_pass(x_i)
            l_art = ((y_i - lp_x).abs() * A).mean()
        else:
            l_art = torch.zeros_like(l_recon)

        # 3) Edge preservation (only in non-artifact regions to avoid conflict)
        grad_y = self.sobel_gradient(y_i)
        grad_x = self.sobel_gradient(x_i)
        l_edge = F.mse_loss(grad_y * non_artifact, grad_x * non_artifact)

        # 4) High-frequency regularization
        hf = self.high_freq(y_i)
        l_hf = hf.abs().mean()

        # 5) Noise level control
        num_body = int(body_mask.sum().item())

        if num_body >= self.min_body_pixels:
            x_body = x_i * body_mask
            y_body = y_i * body_mask

            x_vals = x_body[body_mask > 0.5]
            y_vals = y_body[body_mask > 0.5]

            sigma_in = x_vals.std()
            sigma_out = y_vals.std()

            target = self.target_noise_ratio * sigma_in.detach()
            l_noise = (sigma_out - target).pow(2)
        else:
            l_noise = torch.zeros_like(l_recon)

        # Total loss
        total = (
            l_recon +
            self.lambda_rc * l_rc +
            self.lambda_edge * l_edge +
            self.lambda_hf * l_hf +
            self.lambda_noise * l_noise +
            self.lambda_ic * l_art
        )

        log = {
            "recon": float(l_recon.item()),
            "rc": float(l_rc.item()),
            "ic": float(l_art.item()),
            "noise": float(l_noise.item()),
            "edge": float(l_edge.item()),
            "hf": float(l_hf.item()),
            "total": float(total.item()),
        }
        return total, log