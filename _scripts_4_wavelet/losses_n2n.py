# losses_n2n.py - Neighbor2Neighbor + Wavelet Sparsity (Adaptive + Edge Preservation)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class PyTorchDWT2D(nn.Module):
    """
    PyTorch-based 2D Discrete Wavelet Transform (Haar)
    100% GPU computation - NO CPU transfers
    """
    
    def __init__(self, wavelet='haar'):
        super().__init__()
        
        if wavelet == 'haar':
            h0 = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.float32)
            h1 = torch.tensor([1/np.sqrt(2), -1/np.sqrt(2)], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Wavelet '{wavelet}' not implemented.")
        
        self.register_buffer('h0', h0)
        self.register_buffer('h1', h1)
    
    def forward(self, x):
        """Single-level 2D DWT"""
        B, C, H, W = x.shape
        
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
            W += 1
        
        h0_2d_row = self.h0.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        h0_2d_col = self.h0.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        h1_2d_row = self.h1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        h1_2d_col = self.h1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        
        x_l = F.conv2d(x, h0_2d_row, stride=(1, 2), padding=(0, 0), groups=C)
        x_h = F.conv2d(x, h1_2d_row, stride=(1, 2), padding=(0, 0), groups=C)
        
        LL = F.conv2d(x_l, h0_2d_col, stride=(2, 1), padding=(0, 0), groups=C)
        LH = F.conv2d(x_l, h1_2d_col, stride=(2, 1), padding=(0, 0), groups=C)
        HL = F.conv2d(x_h, h0_2d_col, stride=(2, 1), padding=(0, 0), groups=C)
        HH = F.conv2d(x_h, h1_2d_col, stride=(2, 1), padding=(0, 0), groups=C)
        
        return LL, LH, HL, HH
    
    def multi_level(self, x, levels):
        """Multi-level DWT"""
        coeffs = []
        current = x
        
        for _ in range(levels):
            LL, LH, HL, HH = self.forward(current)
            coeffs.append((LH, HL, HH))
            current = LL
        
        return current, coeffs


class Neighbor2NeighborLoss(nn.Module):
    """
    Neighbor2Neighbor Loss (CVPR 2021) - EXACT Implementation
    """
    
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        print(f"\nNeighbor2Neighbor Loss:")
        print(f"   gamma = {gamma}")
        print(f"   Loss = L_rec + {gamma} * L_reg")
    
    def generate_subimages_checkerboard(self, noisy):
        """Generate two spatially-disjoint sub-images"""
        B, C, H, W = noisy.shape
        
        if H % 2 != 0:
            noisy = noisy[:, :, :-1, :]
            H = H - 1
        if W % 2 != 0:
            noisy = noisy[:, :, :, :-1]
            W = W - 1
        
        pos_0 = noisy[:, :, 0::2, 0::2]
        pos_3 = noisy[:, :, 1::2, 1::2]
        
        g1 = F.interpolate(pos_0, size=(H, W), mode='bilinear', align_corners=False)
        g2 = F.interpolate(pos_3, size=(H, W), mode='bilinear', align_corners=False)
        
        return g1, g2
    
    def forward(self, model, noisy_input, return_output=False):
        """Compute N2N loss"""
        g1, g2 = self.generate_subimages_checkerboard(noisy_input)
        
        output = model(g1)
        output = torch.clamp(output, 0, 1)
        
        rec_loss = F.mse_loss(output, g2)
        reg_loss = F.mse_loss(output, g1)
        
        total = rec_loss + self.gamma * reg_loss
        
        loss_dict = {
            'rec': rec_loss.item(),
            'reg': reg_loss.item(),
            'reg_weighted': (self.gamma * reg_loss).item(),
            'total': total.item()
        }
        
        if return_output:
            return total, loss_dict, output, g1
        else:
            return total, loss_dict


class WaveletSparsityPrior(nn.Module):
    """
    Wavelet Sparsity Prior - FIXED: Estimate noise from INPUT, not output
    """
    
    def __init__(self, threshold=60, wavelet='haar', levels=3, hu_window=(-160, 240), adaptive=True):
        super().__init__()
        
        self.hu_range = hu_window[1] - hu_window[0]
        self.base_threshold = threshold / self.hu_range
        self.base_threshold_hu = threshold
        self.levels = levels
        self.adaptive = adaptive
        
        self.dwt = PyTorchDWT2D(wavelet=wavelet)
        
        print(f"\nWavelet Sparsity Prior (FIXED Adaptive):")
        print(f"   Base Threshold: {threshold} HU -> {self.base_threshold:.4f}")
        print(f"   Levels: {levels}")
        print(f"   ✅ Noise estimated from INPUT (not output)")
    
    def estimate_noise_from_input(self, noisy_input):
        """
        Estimate noise level from NOISY INPUT using MAD
        This is the CORRECT way - estimate from input, not denoised output
        """
        _, detail_coeffs_list = self.dwt.multi_level(noisy_input, 1)
        
        if len(detail_coeffs_list) > 0:
            LH, HL, HH = detail_coeffs_list[0]
            
            all_details = torch.cat([
                LH.flatten(1),
                HL.flatten(1),
                HH.flatten(1)
            ], dim=1)
            
            mad = torch.median(torch.abs(all_details), dim=1)[0]
            sigma = mad / 0.6745
            
            return sigma
        
        return torch.zeros(noisy_input.size(0), device=noisy_input.device)
    
    def soft_threshold(self, coeffs, threshold):
        """Soft Thresholding Operator"""
        return torch.sign(coeffs) * torch.clamp(torch.abs(coeffs) - threshold, min=0)
    
    def forward(self, pred, estimated_sigma=None):
        """
        Compute wavelet sparsity loss
        estimated_sigma: pre-computed noise level from input
        """
        B, C, H, W = pred.shape
        device = pred.device
        
        LL_final, detail_coeffs_list = self.dwt.multi_level(pred, self.levels)
        
        if self.adaptive and estimated_sigma is not None:
            adaptive_threshold = torch.clamp(
                estimated_sigma * 2.5,
                min=self.base_threshold * 0.3,
                max=self.base_threshold * 3.0
            )
            adaptive_threshold = adaptive_threshold.view(B, 1, 1, 1)
        else:
            adaptive_threshold = self.base_threshold
            estimated_sigma = torch.zeros(B, device=device)
        
        total_loss = 0.0
        
        for level_idx, (LH, HL, HH) in enumerate(detail_coeffs_list, start=1):
            level_threshold = adaptive_threshold / (2 ** (level_idx - 1))
            
            LH_sparse = self.soft_threshold(LH, level_threshold).detach()
            HL_sparse = self.soft_threshold(HL, level_threshold).detach()
            HH_sparse = self.soft_threshold(HH, level_threshold).detach()
            
            loss_lh = F.l1_loss(LH, LH_sparse)
            loss_hl = F.l1_loss(HL, HL_sparse)
            loss_hh = F.l1_loss(HH, HH_sparse)
            
            level_weight = 1.0 / level_idx
            total_loss = total_loss + level_weight * (loss_lh + loss_hl + loss_hh) / 3.0
        
        avg_sigma = estimated_sigma.mean().item() if isinstance(estimated_sigma, torch.Tensor) else 0.0
        
        return total_loss, avg_sigma


class EdgePreservationLoss(nn.Module):
    """
    Edge Preservation Loss - Prevents blurring at edges
    Uses Sobel operator to detect edges and penalize changes
    """
    
    def __init__(self, edge_weight=0.1):
        super().__init__()
        self.edge_weight = edge_weight
        
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        
        print(f"\nEdge Preservation Loss:")
        print(f"   Edge weight: {edge_weight}")
        print(f"   ✅ Sobel-based edge detection")
    
    def get_edges(self, x):
        """Extract edges using Sobel operator"""
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        return edges
    
    def forward(self, noisy_input, denoised_output):
        """
        Compute edge preservation loss
        Penalize edge changes between input and output
        """
        input_edges = self.get_edges(noisy_input)
        output_edges = self.get_edges(denoised_output)
        
        # Edge magnitude preservation
        edge_diff = torch.abs(input_edges - output_edges)
        
        # Weight by edge strength (preserve strong edges more)
        edge_weight_map = torch.clamp(input_edges / (input_edges.mean() + 1e-8), 0, 3)
        weighted_diff = edge_diff * edge_weight_map
        
        return weighted_diff.mean()


class CombinedN2NWaveletLoss(nn.Module):
    """
    Combined Loss: N2N + Wavelet + Edge Preservation
    
    - Noise는 INPUT에서 추정
    - Wavelet weight를 샘플별(HN/LN별) adaptive하게 적용
    """

    def __init__(self,
                 n2n_gamma=2.0,
                 wavelet_weight=0.0025,
                 wavelet_threshold=60,
                 wavelet_levels=3,
                 hu_window=(-160, 240),
                 adaptive=True,
                 # 🔧 Step2: HN/LN 구분을 위해 target_noise/범위를 조금 더 보수적으로 설정
                 target_noise=0.012,                # 대략 LN σ 근처 (정확한 값은 데이터에 따라 조정)
                 adaptive_weight_range=(0.8, 4.0),  # LN ~0.8x, HN 최대 ~4x
                 edge_weight=0.08):                 # edge 보존 조금 더 강하게
        super().__init__()

        self.base_wavelet_weight = wavelet_weight
        self.target_noise = target_noise
        self.adaptive = adaptive
        self.weight_min, self.weight_max = adaptive_weight_range
        self.edge_weight = edge_weight

        self.n2n_loss = Neighbor2NeighborLoss(gamma=n2n_gamma)

        self.wavelet_loss = WaveletSparsityPrior(
            threshold=wavelet_threshold,
            wavelet='haar',
            levels=wavelet_levels,
            hu_window=hu_window,
            adaptive=adaptive
        )

        self.edge_loss = EdgePreservationLoss(edge_weight=edge_weight)

        print("\n🔧 Combined Loss (Sample-wise Adaptive + Edge Preservation)")
        print(f"   N2N gamma        : {n2n_gamma}")
        print(f"   Base wavelet w   : {wavelet_weight}")
        print(f"   Target noise     : {target_noise:.4f} (norm.)")
        print(f"   Weight range     : [{self.weight_min:.2f}, {self.weight_max:.2f}]")
        print(f"   Edge weight      : {edge_weight}")

    def forward(self, model, noisy_input):
        """
        model: SwinIR
        noisy_input: [B, 1, H, W]  (normalized)
        """
        B = noisy_input.size(0)

        # 1) INPUT 기준 noise estimate (샘플별 σ_i)
        estimated_sigma = self.wavelet_loss.estimate_noise_from_input(noisy_input)  # [B]

        # 2) N2N loss + output
        n2n_total, n2n_dict, output, g1 = self.n2n_loss(model, noisy_input, return_output=True)

        # 3) Wavelet sparsity (샘플별 loss, 샘플별 weight)
        per_sample_wavelet = []
        per_sample_weight = []
        per_sample_ratio = []

        for i in range(B):
            # 각 샘플별 wavelet loss (σ_i에 맞는 threshold 사용)
            w_loss_i, _ = self.wavelet_loss(
                output[i:i+1],           # [1,1,H,W]
                estimated_sigma[i:i+1]   # [1]
            )
            per_sample_wavelet.append(w_loss_i)

            if self.adaptive and self.target_noise > 0:
                ratio_i = (estimated_sigma[i] / self.target_noise).clamp(
                    self.weight_min, self.weight_max
                )
                weight_i = self.base_wavelet_weight * ratio_i
            else:
                ratio_i = torch.tensor(1.0, device=noisy_input.device)
                weight_i = torch.tensor(self.base_wavelet_weight, device=noisy_input.device)

            per_sample_weight.append(weight_i)
            per_sample_ratio.append(ratio_i)

        per_sample_wavelet = torch.stack(per_sample_wavelet)   # [B]
        per_sample_weight = torch.stack(per_sample_weight)     # [B]
        per_sample_ratio = torch.stack(per_sample_ratio)       # [B]

        # 🔹 HN/LN별 weight 차이를 실제 loss에 반영
        #   (가중 평균으로 배치 전체 wavelet loss 구성)
        weighted_wavelet = per_sample_weight * per_sample_wavelet   # [B]
        wavelet = 2.0*weighted_wavelet.mean()                           # scalar
        wavelet_raw = per_sample_wavelet.mean()                     # unweighted 평균

        # 4) Edge preservation loss (여전히 전체 배치 기준)
        edge = self.edge_loss(g1, output)

        # 5) Total loss
        total = n2n_total + wavelet + self.edge_weight * edge

        # NaN 보호
        if torch.isnan(total):
            total = n2n_total
            wavelet = torch.tensor(0.0, device=noisy_input.device)
            edge = torch.tensor(0.0, device=noisy_input.device)

        # 모니터링용 통계 (기존 필드 유지 + 몇 개 추가)
        avg_sigma = float(estimated_sigma.mean().item())
        avg_sigma_hu = avg_sigma * self.wavelet_loss.hu_range if hasattr(self.wavelet_loss, 'hu_range') else avg_sigma * 400
        avg_weight = float(per_sample_weight.mean().item())
        avg_ratio = float(per_sample_ratio.mean().item())

        return total, {
            'n2n_rec': n2n_dict['rec'],
            'n2n_reg': n2n_dict['reg'],
            'n2n_reg_weighted': n2n_dict['reg_weighted'],
            'n2n_total': n2n_dict['total'],
            'wavelet_raw': float(wavelet_raw.item()),
            'wavelet_weighted': float(wavelet.item()),
            'edge_loss': float(edge.item()),
            'total': float(total.item()),
            'balance_ratio': n2n_dict['total'] / (wavelet.item() + 1e-8),
            'estimated_noise': avg_sigma,
            'estimated_noise_hu': avg_sigma_hu,
            'adaptive_weight': avg_weight,   # utils.save_sample_images에서 사용
            'noise_ratio': avg_ratio
        }


# Backward compatibility
Neighbor2NeighborLoss_v2 = Neighbor2NeighborLoss
WaveletSparsityLoss = WaveletSparsityPrior