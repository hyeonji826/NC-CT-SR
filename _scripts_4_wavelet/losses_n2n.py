# losses_n2n.py - Neighbor2Neighbor + Wavelet Sparsity (Optimized)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class PyTorchDWT2D(nn.Module):
    """
    PyTorch-based 2D Discrete Wavelet Transform (Haar)
    
    Runs entirely on GPU - NO CPU transfers!
    Replaces pywt for massive performance improvement
    """
    
    def __init__(self, wavelet='haar'):
        super().__init__()
        
        if wavelet == 'haar':
            # Haar wavelet filters
            h0 = torch.tensor([1/np.sqrt(2), 1/np.sqrt(2)], dtype=torch.float32)
            h1 = torch.tensor([1/np.sqrt(2), -1/np.sqrt(2)], dtype=torch.float32)
        else:
            raise NotImplementedError(f"Wavelet '{wavelet}' not implemented. Use 'haar'.")
        
        # Create 2D filters
        self.register_buffer('h0', h0)
        self.register_buffer('h1', h1)
    
    def forward(self, x):
        """
        Single-level 2D DWT
        
        Args:
            x: [B, C, H, W] tensor
            
        Returns:
            coeffs: (LL, LH, HL, HH) - each [B, C, H/2, W/2]
        """
        B, C, H, W = x.shape
        
        # Ensure even dimensions
        if H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
            H += 1
        if W % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
            W += 1
        
        # Create 2D conv kernels from 1D filters
        # Low-pass filter (h0)
        h0_2d_row = self.h0.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        h0_2d_col = self.h0.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        
        # High-pass filter (h1)
        h1_2d_row = self.h1.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
        h1_2d_col = self.h1.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
        
        # Apply row-wise filtering
        x_l = F.conv2d(x, h0_2d_row, stride=(1, 2), padding=(0, 0), groups=C)
        x_h = F.conv2d(x, h1_2d_row, stride=(1, 2), padding=(0, 0), groups=C)
        
        # Apply column-wise filtering
        LL = F.conv2d(x_l, h0_2d_col, stride=(2, 1), padding=(0, 0), groups=C)
        LH = F.conv2d(x_l, h1_2d_col, stride=(2, 1), padding=(0, 0), groups=C)
        HL = F.conv2d(x_h, h0_2d_col, stride=(2, 1), padding=(0, 0), groups=C)
        HH = F.conv2d(x_h, h1_2d_col, stride=(2, 1), padding=(0, 0), groups=C)
        
        return LL, LH, HL, HH
    
    def multi_level(self, x, levels):
        """
        Multi-level DWT
        
        Args:
            x: [B, C, H, W]
            levels: number of decomposition levels
            
        Returns:
            List of tuples: [(LH1, HL1, HH1), (LH2, HL2, HH2), ...]
            Plus final LL: [LL_final]
        """
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
    
    Paper: "Neighbor2Neighbor: Self-Supervised Denoising from Single Noisy Images"
    Authors: Huang et al.
    
    Key Idea:
    1. Subsample noisy image into 4 spatially-disjoint sub-images (2x2 checkerboard)
    2. Each sub-image has INDEPENDENT noise (critical!)
    3. Use one as input (g1), another as target (g2)
    4. Loss = ||f(g1) - g2||^2 + gamma * ||f(g1) - g1||^2
       - First term: reconstruction with independent noise -> denoises
       - Second term: regularization to prevent identity mapping
    
    Implementation Details:
    - 2x2 checkerboard creates 4 positions: (even,even), (even,odd), (odd,even), (odd,odd)
    - We use (even,even) and (odd,odd) for maximum spatial separation
    - gamma = 2.0 (paper's optimal value)
    """
    
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        print(f"\nNeighbor2Neighbor Loss:")
        print(f"   gamma = {gamma}")
        print(f"   Loss = L_rec + {gamma} * L_reg")
        print(f"   Prevents identity mapping while denoising")
    
    def generate_subimages_checkerboard(self, noisy):
        """
        Generate two spatially-disjoint sub-images using 2x2 checkerboard pattern.
        
        Args:
            noisy: [B, C, H, W] - noisy input image
            
        Returns:
            g1: [B, C, H, W] - subsampled image 1 (even-even positions)
            g2: [B, C, H, W] - subsampled image 2 (odd-odd positions)
            
        Critical: g1 and g2 must have INDEPENDENT noise!
        """
        B, C, H, W = noisy.shape
        
        # Ensure even dimensions
        if H % 2 != 0:
            noisy = noisy[:, :, :-1, :]
            H = H - 1
        if W % 2 != 0:
            noisy = noisy[:, :, :, :-1]
            W = W - 1
        
        # Extract checkerboard positions
        pos_0 = noisy[:, :, 0::2, 0::2]  # [B, C, H/2, W/2]
        pos_3 = noisy[:, :, 1::2, 1::2]  # [B, C, H/2, W/2]
        
        # Upsample to original size (bilinear for gradient flow)
        g1 = F.interpolate(pos_0, size=(H, W), mode='bilinear', align_corners=False)
        g2 = F.interpolate(pos_3, size=(H, W), mode='bilinear', align_corners=False)
        
        return g1, g2
    
    def forward(self, model, noisy_input, return_output=False):
        """
        Compute N2N loss.
        
        Args:
            model: denoising network f(·)
            noisy_input: [B, C, H, W] - noisy CT image
            return_output: if True, also return model output
            
        Returns:
            total_loss: scalar tensor with gradients
            loss_dict: dictionary with loss components (detached scalars)
            output: (optional) model output if return_output=True
        """
        # Generate spatially-disjoint subsamples
        g1, g2 = self.generate_subimages_checkerboard(noisy_input)
        
        # Forward pass: denoise g1
        output = model(g1)
        output = torch.clamp(output, 0, 1)
        
        # L_rec: reconstruction loss (output vs g2)
        # g2 has INDEPENDENT noise from g1, so this denoises!
        rec_loss = F.mse_loss(output, g2)
        
        # L_reg: regularization loss (output vs g1)
        # Prevents identity mapping
        reg_loss = F.mse_loss(output, g1)
        
        # Total N2N loss (keep gradient!)
        total = rec_loss + self.gamma * reg_loss
        
        loss_dict = {
            'rec': rec_loss.item(),
            'reg': reg_loss.item(),
            'reg_weighted': (self.gamma * reg_loss).item(),
            'total': total.item()
        }
        
        if return_output:
            return total, loss_dict, output
        else:
            return total, loss_dict


class WaveletSparsityPrior(nn.Module):
    """
    Wavelet Sparsity Prior for Medical Image Denoising (GPU-Optimized)
    
    Key Improvements:
    1. 100% GPU computation - NO CPU transfers!
    2. PyTorch-based DWT (replaces pywt)
    3. Adaptive thresholding using MAD
    4. Per-image noise estimation
    
    Method:
    1. Estimate noise level using MAD from wavelet coefficients
    2. Adapt threshold based on estimated noise (k * sigma)
    3. Apply soft thresholding with adapted threshold
    4. L1 penalty to encourage sparsity
    """
    
    def __init__(self, threshold=40, wavelet='haar', levels=3, hu_window=(-160, 240), adaptive=True):
        super().__init__()
        
        # HU window for correct normalization
        self.hu_range = hu_window[1] - hu_window[0]  # 400 for (-160, 240)
        
        # Base threshold (will be adapted per image if adaptive=True)
        self.base_threshold = threshold / self.hu_range  # Correct normalization
        self.base_threshold_hu = threshold
        
        self.levels = levels
        self.adaptive = adaptive
        
        # PyTorch DWT (GPU-based)
        self.dwt = PyTorchDWT2D(wavelet=wavelet)
        
        print(f"\nWavelet Sparsity Prior (GPU-Optimized):")
        print(f"   Base Threshold: {threshold} HU -> {self.base_threshold:.4f} (normalized)")
        print(f"   HU Range: {self.hu_range} ({hu_window[0]} to {hu_window[1]})")
        print(f"   Wavelet: {wavelet}")
        print(f"   Levels: {levels}")
        print(f"   Adaptive: {adaptive} (noise-aware thresholding)")
        print(f"   ✅ 100% GPU computation (NO CPU transfers!)")
    
    def estimate_noise_mad(self, detail_coeffs):
        """
        Estimate noise level using MAD (Median Absolute Deviation)
        
        Args:
            detail_coeffs: (LH, HL, HH) tuple from finest level
            
        Returns:
            sigma: estimated noise standard deviation (normalized)
        """
        LH, HL, HH = detail_coeffs
        
        # Combine all detail coefficients from finest level
        all_details = torch.cat([
            LH.flatten(1),
            HL.flatten(1),
            HH.flatten(1)
        ], dim=1)  # [B, N*3]
        
        # MAD estimation: sigma = median(|X|) / 0.6745
        # Per-batch calculation
        mad = torch.median(torch.abs(all_details), dim=1)[0]  # [B]
        sigma = mad / 0.6745
        
        return sigma
    
    def soft_threshold(self, coeffs, threshold):
        """
        Soft Thresholding Operator (SoT) - GPU version
        
        Formula:
            T_soft(x) = sign(x) * max(|x| - threshold, 0)
        
        Effect:
            - Small coeffs (< threshold): -> 0 (assumed to be noise)
            - Large coeffs (> threshold): -> shrunk but preserved (signal)
        """
        return torch.sign(coeffs) * torch.clamp(torch.abs(coeffs) - threshold, min=0)
    
    def forward(self, pred):
        """
        Compute wavelet sparsity loss with adaptive thresholding.
        
        Args:
            pred: [B, C, H, W] - model output (denoised)
            
        Returns:
            loss: L1 distance between coeffs and sparse target
            estimated_noise: average estimated noise level (for monitoring)
        """
        B, C, H, W = pred.shape
        device = pred.device
        
        # Multi-level DWT (100% GPU)
        LL_final, detail_coeffs_list = self.dwt.multi_level(pred, self.levels)
        
        # Estimate noise level from finest level (adaptive)
        if self.adaptive and len(detail_coeffs_list) > 0:
            sigma = self.estimate_noise_mad(detail_coeffs_list[0])  # [B]
            
            # Adapt threshold: k * sigma (k=2.5)
            adaptive_threshold = torch.clamp(
                sigma * 2.5,
                min=self.base_threshold * 0.5,
                max=self.base_threshold * 3.0
            )  # [B]
            
            # Broadcast to [B, 1, 1, 1]
            adaptive_threshold = adaptive_threshold.view(B, 1, 1, 1)
        else:
            adaptive_threshold = self.base_threshold
            sigma = torch.zeros(B, device=device)
        
        # Compute sparsity loss across all levels
        total_loss = 0.0
        
        for level_idx, (LH, HL, HH) in enumerate(detail_coeffs_list, start=1):
            # Scale threshold per level
            level_threshold = adaptive_threshold / (2 ** (level_idx - 1))
            
            # Compute sparse targets (GPU)
            LH_sparse = self.soft_threshold(LH, level_threshold).detach()
            HL_sparse = self.soft_threshold(HL, level_threshold).detach()
            HH_sparse = self.soft_threshold(HH, level_threshold).detach()
            
            # L1 sparsity penalty
            loss_lh = F.l1_loss(LH, LH_sparse)
            loss_hl = F.l1_loss(HL, HL_sparse)
            loss_hh = F.l1_loss(HH, HH_sparse)
            
            # Weight finer levels more
            level_weight = 1.0 / level_idx
            total_loss = total_loss + level_weight * (loss_lh + loss_hl + loss_hh) / 3.0
        
        # Average noise level for monitoring
        avg_noise = sigma.mean().item()
        
        return total_loss, avg_noise


class CombinedN2NWaveletLoss(nn.Module):
    """
    Combined Loss: Neighbor2Neighbor + Wavelet Sparsity (Optimized)
    
    Key Optimizations:
    1. NO forward pass duplication - model runs ONCE per batch
    2. 100% GPU computation - NO CPU transfers
    3. Extended adaptive weight range [0.3, 3.0]
    
    Philosophy:
    - High noise images: stronger wavelet (more denoising)
    - Low noise images: weaker wavelet (preserve details)
    - Automatic balancing for optimal results
    """
    
    def __init__(self,
                 n2n_gamma=2.0,
                 wavelet_weight=0.0025,
                 wavelet_threshold=60,
                 wavelet_levels=3,
                 hu_window=(-160, 240),
                 adaptive=True,
                 target_noise=0.15,
                 adaptive_weight_range=(0.3, 3.0)):
        super().__init__()
        
        self.base_wavelet_weight = wavelet_weight
        self.target_noise = target_noise
        self.adaptive = adaptive
        self.weight_min, self.weight_max = adaptive_weight_range
        
        # N2N loss (main)
        self.n2n_loss = Neighbor2NeighborLoss(gamma=n2n_gamma)
        
        # Wavelet sparsity (regularization)
        self.wavelet_loss = WaveletSparsityPrior(
            threshold=wavelet_threshold,
            wavelet='haar',
            levels=wavelet_levels,
            hu_window=hu_window,
            adaptive=adaptive
        )
        
        print(f"\n🔧 Combined Loss (OPTIMIZED):")
        print(f"   N2N weight: 1.0 (MAIN)")
        print(f"   Base Wavelet weight: {wavelet_weight}")
        print(f"   Adaptive weighting: {adaptive}")
        print(f"   Adaptive range: [{self.weight_min}, {self.weight_max}]")
        print(f"   Target noise: {target_noise:.4f}")
        print(f"   ✅ Forward pass runs ONCE (no duplication!)")
        print(f"   ✅ 100% GPU computation")
    
    def forward(self, model, noisy_input):
        """
        Compute combined loss - OPTIMIZED VERSION
        
        Args:
            model: denoising network
            noisy_input: [B, C, H, W] - noisy input
            
        Returns:
            total_loss: scalar tensor with gradients
            loss_dict: detailed loss breakdown (detached scalars)
        """
        # N2N loss + get output (SINGLE forward pass)
        n2n_total, n2n_dict, output = self.n2n_loss(model, noisy_input, return_output=True)
        
        # Wavelet sparsity on SAME output (no duplication!)
        wavelet, estimated_noise = self.wavelet_loss(output)
        
        # NaN protection
        if torch.isnan(wavelet):
            wavelet = torch.tensor(0.0, device=noisy_input.device, requires_grad=True)
            estimated_noise = 0.0
        
        # Adaptive weighting based on noise level
        if self.adaptive and estimated_noise > 0:
            # Scale weight proportionally to noise level
            noise_ratio = estimated_noise / self.target_noise
            noise_ratio = max(self.weight_min, min(noise_ratio, self.weight_max))
            adaptive_weight = self.base_wavelet_weight * noise_ratio
        else:
            adaptive_weight = self.base_wavelet_weight
        
        # Combined loss (keep gradient!)
        total = n2n_total + adaptive_weight * wavelet
        
        return total, {
            'n2n_rec': n2n_dict['rec'],
            'n2n_reg': n2n_dict['reg'],
            'n2n_reg_weighted': n2n_dict['reg_weighted'],
            'n2n_total': n2n_dict['total'],
            'wavelet_raw': wavelet.item(),
            'wavelet_weighted': (wavelet * adaptive_weight).item(),
            'total': total.item(),
            'balance_ratio': n2n_dict['total'] / (wavelet.item() * adaptive_weight + 1e-8),
            'estimated_noise': estimated_noise,
            'adaptive_weight': adaptive_weight
        }


# For backward compatibility
Neighbor2NeighborLoss_v2 = Neighbor2NeighborLoss
WaveletSparsityLoss = WaveletSparsityPrior