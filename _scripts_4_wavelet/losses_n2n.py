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
      1. L_flat  : Aggressive denoising in flat regions - (y_i ≈ x_{i+1}) * W * M_flat
      2. L_rc    : Self-consistency - y_i ≈ x_i (content preservation in matched regions)
      3. L_edge  : Edge structure preservation - gradient similarity in NON-ARTIFACT regions
      4. L_hf    : High-frequency regularization - prevents over-sharpening and residual noise
      5. L_noise : Noise level control - body region std(y_i) ≈ target_ratio * std(x_i)
      6. L_art   : Artifact suppression - selective smoothing of gradient outlier regions
    
    Key Design Decision (FLAT vs EDGE separation):
      - M_flat: Low gradient regions (< threshold) → aggressive noise removal
      - M_edge: High gradient regions (≥ threshold) → structure preservation priority
      - This addresses the core NS-N2N assumption violation: adjacent slices differ at edges
      - Flat regions: noise only (same structure) → strong denoising
      - Edge regions: structure change + noise → conservative, preserve boundaries
    
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
        flat_threshold: float = 0.05,
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
        self.flat_threshold = float(flat_threshold)

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
        Forward pass with FLAT vs EDGE separation
        
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

        # Compute gradient magnitude for flat/edge separation
        grad_x = self.sobel_gradient(x_i)
        M_flat = (grad_x < self.flat_threshold).float()
        M_edge = (grad_x >= self.flat_threshold).float()

        # 1) L_flat: Aggressive denoising in flat regions (NS-N2N principle)
        # Flat regions have minimal structure change between slices → noise dominates
        diff_flat = (y_i - x_ip1) * W * M_flat
        l_flat = diff_flat.pow(2).mean()

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
            l_art = torch.zeros_like(l_flat)

        # 3) Edge preservation (only in non-artifact AND edge regions)
        # Edge regions: structure preservation is priority, noise removal is secondary
        grad_y = self.sobel_gradient(y_i)
        edge_protection_mask = non_artifact * M_edge
        l_edge = F.mse_loss(grad_y * edge_protection_mask, grad_x * edge_protection_mask)

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
            l_noise = torch.zeros_like(l_flat)

        # Total loss (L_recon replaced by L_flat)
        total = (
            l_flat +
            self.lambda_rc * l_rc +
            self.lambda_edge * l_edge +
            self.lambda_hf * l_hf +
            self.lambda_noise * l_noise +
            self.lambda_ic * l_art
        )

        log = {
            "flat": float(l_flat.item()),
            "rc": float(l_rc.item()),
            "ic": float(l_art.item()),
            "noise": float(l_noise.item()),
            "edge": float(l_edge.item()),
            "hf": float(l_hf.item()),
            "total": float(total.item()),
            "flat_ratio": float(M_flat.mean().item()),
            "edge_ratio": float(M_edge.mean().item()),
        }
        return total, log