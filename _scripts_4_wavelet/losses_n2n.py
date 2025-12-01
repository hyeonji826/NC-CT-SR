# E:\LD-CT SR\_scripts_4_wavelet\losses_n2n.py
# NA-NSN2N Loss: NS-N2N + Noise Residual Learning

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NSN2NLoss(nn.Module):
    """
    Noise Augmented NS-N2N Loss
    
    Combines:
    1. NS-N2N reconstruction (matched region)
    2. Noise residual matching (synthetic noise)
    3. Edge preservation
    4. High-frequency suppression
    """

    def __init__(
        self, 
        lambda_rc: float = 1.5,
        lambda_ic: float = 3.0,
        lambda_noise: float = 2.0,
        lambda_edge: float = 0.3,
        lambda_hf: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_rc = float(lambda_rc)
        self.lambda_ic = float(lambda_ic)
        self.lambda_noise = float(lambda_noise)
        self.lambda_edge = float(lambda_edge)
        self.lambda_hf = float(lambda_hf)

    def sobel_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Sobel gradient magnitude"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return grad

    def forward(
        self,
        noise_pred: torch.Tensor,
        x_i: torch.Tensor,              # ⭐ 원본 추가
        x_i_aug: torch.Tensor,
        x_ip1: torch.Tensor,
        x_mid: torch.Tensor,
        W: torch.Tensor,
        noise_synthetic: torch.Tensor,
    ):
        W = W.detach()
        
        # Denoised outputs
        y_i = x_i_aug - noise_pred
        y_ip1 = x_ip1  # Next slice as is (could also denoise)
        y_mid = x_mid  # Middle slice as is

        # 1) NS-N2N Reconstruction (matched region)
        diff1 = (y_i - x_ip1) * W
        l_recon = diff1.pow(2).mean()

        # 2) Regional consistency
        l_rc = ((y_i - y_ip1) * W).pow(2).mean()

        # 3) Inter-slice continuity
        target_mid = 0.5 * (y_i + y_ip1)
        l_ic = (y_mid - target_mid).pow(2).mean()

        # 4) Noise residual matching (핵심!)
        l_noise = F.mse_loss(noise_pred, noise_synthetic)

        # 5) Edge preservation - 원본과 비교
        grad_original = self.sobel_gradient(x_i)  # ⭐ 원본
        grad_output = self.sobel_gradient(y_i)
        l_edge = F.mse_loss(grad_output, grad_original)

        # 6) High-frequency suppression
        hf_i = y_i - F.avg_pool2d(
            F.pad(y_i, (2,2,2,2), mode='reflect'),
            kernel_size=5, stride=1
        )
        l_hf = hf_i.abs().mean()

        total = (
            l_recon +
            self.lambda_rc * l_rc +
            self.lambda_ic * l_ic +
            self.lambda_noise * l_noise +
            self.lambda_edge * l_edge +
            self.lambda_hf * l_hf
        )

        log = {
            "recon": float(l_recon.item()),
            "rc": float(l_rc.item()),
            "ic": float(l_ic.item()),
            "noise": float(l_noise.item()),
            "edge": float(l_edge.item()),
            "hf": float(l_hf.item()),
            "total": float(total.item()),
        }
        return total, log