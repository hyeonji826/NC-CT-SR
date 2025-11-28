# E:\LD-CT SR\_scripts_4_wavelet\losses_n2n.py
# NS-N2N core loss (reconstruction + regional consistency + inter-slice continuity)

from __future__ import annotations

import torch
import torch.nn as nn


class NSN2NLoss(nn.Module):
    """
    NS-N2N self-supervised loss.
      y_i, y_ip1, y_mid : [B,1,H,W] (network outputs)
      x_i, x_ip1        : [B,1,H,W] (noisy inputs, 0~1)
      W                 : [B,1,H,W] (0/1 matched region mask)
    """

    def __init__(self, lambda_rc: float = 0.5, lambda_ic: float = 1.0) -> None:
        super().__init__()
        self.lambda_rc = float(lambda_rc)
        self.lambda_ic = float(lambda_ic)

    def forward(
        self,
        y_i: torch.Tensor,
        y_ip1: torch.Tensor,
        y_mid: torch.Tensor,
        x_i: torch.Tensor,
        x_ip1: torch.Tensor,
        W: torch.Tensor,
    ):
        W = W.detach()

        # 1) Reconstruction (cross Neighbor2Neighbor, matched 영역만)
        diff1 = (y_i - x_ip1) * W
        diff2 = (y_ip1 - x_i) * W
        l_recon = 0.5 * (diff1.pow(2).mean() + diff2.pow(2).mean())

        # 2) Regional consistency (matched 영역에서 출력 둘이 같도록)
        l_rc = ((y_i - y_ip1) * W).pow(2).mean()

        # 3) Inter-slice continuity (중간 슬라이스가 평균에 가깝도록)
        target_mid = 0.5 * (y_i + y_ip1)
        l_ic = (y_mid - target_mid).pow(2).mean()

        total = l_recon + self.lambda_rc * l_rc + self.lambda_ic * l_ic

        log = {
            "recon": float(l_recon.item()),
            "rc": float(l_rc.item()),
            "ic": float(l_ic.item()),
            "total": float(total.item()),
        }
        return total, log
