# E:\LD-CT SR\_scripts_4_wavelet\losses_n2n.py
# Supervised 2.5D Loss for SwinIR (MSE + Wavelet Sparsity)

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 2D Discrete Wavelet Transform (Haar) in PyTorch
# ------------------------------------------------------------
class PyTorchDWT2D(nn.Module):
    """
    Simple 2D Haar DWT implemented in PyTorch.
    Input:  [B, 1, H, W]
    Output: cA, (cH, cV, cD) for each level
    """

    def __init__(self, wave="haar"):
        super().__init__()

        if wave != "haar":
            raise NotImplementedError("Only 'haar' wavelet is implemented.")

        # Haar filters
        h0 = torch.tensor([1.0, 1.0]) / np.sqrt(2.0)  # low-pass
        h1 = torch.tensor([-1.0, 1.0]) / np.sqrt(2.0)  # high-pass

        ll = torch.outer(h0, h0)
        lh = torch.outer(h0, h1)
        hl = torch.outer(h1, h0)
        hh = torch.outer(h1, h1)

        self.register_buffer("ll", ll.view(1, 1, 2, 2))
        self.register_buffer("lh", lh.view(1, 1, 2, 2))
        self.register_buffer("hl", hl.view(1, 1, 2, 2))
        self.register_buffer("hh", hh.view(1, 1, 2, 2))

    def forward(self, x):
        """
        x: [B, 1, H, W]
        """
        ll = F.conv2d(x, self.ll, stride=2)
        lh = F.conv2d(x, self.lh, stride=2)
        hl = F.conv2d(x, self.hl, stride=2)
        hh = F.conv2d(x, self.hh, stride=2)
        return ll, (lh, hl, hh)


# ------------------------------------------------------------
# Wavelet sparsity prior (thresholding high-frequency bands)
# ------------------------------------------------------------
class WaveletSparsityPrior(nn.Module):
    def __init__(
        self,
        threshold=60.0,
        wavelet="haar",
        levels=3,
        hu_window=(-160, 240),
        adaptive=False,
    ):
        """
        threshold: HU 단위 threshold (기본 60 HU)
        adaptive : 여기서는 False로 사용 (고정 threshold)
        """
        super().__init__()
        self.levels = levels
        self.adaptive = adaptive

        self.hu_min, self.hu_max = hu_window
        self.hu_range = self.hu_max - self.hu_min
        self.base_threshold_hu = threshold
        self.base_threshold = threshold / (self.hu_range + 1e-8)

        self.dwt = PyTorchDWT2D(wavelet)

    @torch.no_grad()
    def estimate_noise_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W], 정규화(0~1) 이미지를 HU로 되돌려서 sigma 추정.
        """
        b, c, h, w = x.shape
        assert c == 1, "Noise estimation assumes 1-channel input."

        img_hu = x * self.hu_range + self.hu_min
        flat = img_hu.view(b, -1)
        sigma = flat.std(dim=1) / (self.hu_range + 1e-8)  # 다시 0~1 스케일
        return sigma  # [B]

    def forward(self, x: torch.Tensor, estimated_sigma: torch.Tensor | None = None):
        """
        x: [B, 1, H, W] (0~1 정규화)
        estimated_sigma: 사용하지 않음 (adaptive=False로 운용)
        """
        b, c, h, w = x.shape
        assert c == 1, "Wavelet prior expects 1-channel input."

        # 다중 레벨 DWT
        total_loss = torch.tensor(0.0, device=x.device)
        current = x
        for level in range(1, self.levels + 1):
            ll, (lh, hl, hh) = self.dwt(current)

            # 고주파 계수에 soft-threshold
            thr = self.base_threshold
            lh_sparse = torch.sign(lh) * torch.relu(torch.abs(lh) - thr)
            hl_sparse = torch.sign(hl) * torch.relu(torch.abs(hl) - thr)
            hh_sparse = torch.sign(hh) * torch.relu(torch.abs(hh) - thr)

            loss_lh = F.l1_loss(lh, lh_sparse)
            loss_hl = F.l1_loss(hl, hl_sparse)
            loss_hh = F.l1_loss(hh, hh_sparse)

            level_weight = 1.0 / level
            total_loss = total_loss + level_weight * (loss_lh + loss_hl + loss_hh) / 3.0

            current = ll  # 다음 레벨로

        avg_sigma = (
            float(estimated_sigma.mean().item()) if isinstance(estimated_sigma, torch.Tensor) else 0.0
        )
        return total_loss, avg_sigma


# ------------------------------------------------------------
# Supervised Loss: MSE + λ * Wavelet
# ------------------------------------------------------------
class SupervisedWaveletLoss(nn.Module):
    """
    L = L_MSE(output, target) + λ * L_wavelet(output)

    - output, target: [B, 1, H, W], 0~1
    - Wavelet prior는 output에만 적용
    """

    def __init__(
        self,
        wavelet_weight=0.0025,
        wavelet_threshold=60.0,
        wavelet_levels=3,
        hu_window=(-160, 240),
    ):
        super().__init__()
        self.mse = nn.MSELoss()
        self.wavelet_weight = wavelet_weight
        self.wavelet_loss = WaveletSparsityPrior(
            threshold=wavelet_threshold,
            levels=wavelet_levels,
            hu_window=hu_window,
            adaptive=False,  # 여기서는 adaptive 사용 안 함
        )

        print("\n[SupervisedWaveletLoss]")
        print(f"  wavelet_weight   : {self.wavelet_weight}")
        print(f"  wavelet_threshold: {wavelet_threshold} HU")
        print(f"  wavelet_levels   : {wavelet_levels}")
        print(f"  hu_window        : {hu_window}")
        print("  Loss = MSE + λ * Wavelet\n")

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        """
        output: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        base = self.mse(output, target)

        wavelet = torch.tensor(0.0, device=output.device)
        est_sigma = torch.tensor(0.0, device=output.device)

        if self.wavelet_weight > 0.0:
            wavelet_raw, est_sigma = self.wavelet_loss(output, None)
            wavelet = self.wavelet_weight * wavelet_raw
            total = base + wavelet
        else:
            wavelet_raw = torch.tensor(0.0, device=output.device)
            total = base

        # noise(HU) 추정은 모니터링용
        if isinstance(est_sigma, torch.Tensor):
            sigma_val = float(est_sigma)
        else:
            sigma_val = float(est_sigma)
        sigma_hu = sigma_val * self.wavelet_loss.hu_range

        loss_dict = {
            "base": float(base.item()),
            "wavelet_raw": float(wavelet_raw.item()),
            "wavelet_weighted": float(wavelet.item()),
            "total": float(total.item()),
            "estimated_noise": sigma_val,
            "estimated_noise_hu": sigma_hu,
        }
        return total, loss_dict
