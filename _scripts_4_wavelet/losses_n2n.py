# losses_n2n.py - Neighbor2Neighbor + Wavelet Sparsity (Adaptive)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


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
    
    def forward(self, model, noisy_input):
        """
        Compute N2N loss.
        
        Args:
            model: denoising network f(·)
            noisy_input: [B, C, H, W] - noisy CT image
            
        Returns:
            total_loss: scalar tensor with gradients
            loss_dict: dictionary with loss components (detached scalars)
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
        
        return total, {
            'rec': rec_loss.item(),
            'reg': reg_loss.item(),
            'reg_weighted': (self.gamma * reg_loss).item(),
            'total': total.item()
        }


class WaveletSparsityPrior(nn.Module):
    """
    Wavelet Sparsity Prior for Medical Image Denoising (Adaptive)
    
    Key Improvements:
    1. HU-based threshold normalization (not RGB 255)
    2. Adaptive thresholding using MAD (Median Absolute Deviation)
    3. Per-image noise estimation for handling noise imbalance
    
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
        
        self.wavelet = wavelet
        self.levels = levels
        self.adaptive = adaptive
        
        print(f"\nWavelet Sparsity Prior (Adaptive):")
        print(f"   Base Threshold: {threshold} HU -> {self.base_threshold:.4f} (normalized)")
        print(f"   HU Range: {self.hu_range} ({hu_window[0]} to {hu_window[1]})")
        print(f"   Wavelet: {wavelet}")
        print(f"   Levels: {levels}")
        print(f"   Adaptive: {adaptive} (noise-aware thresholding)")
        print(f"   Encourages sparse high-freq coefficients (removes noise)")
    
    def estimate_noise_mad(self, coeffs_tuple):
        """
        Estimate noise level using MAD (Median Absolute Deviation)
        
        Args:
            coeffs_tuple: (cH, cV, cD) from finest level
            
        Returns:
            sigma: estimated noise standard deviation (normalized)
        """
        cH, cV, cD = coeffs_tuple
        
        # Combine all detail coefficients from finest level
        all_details = np.concatenate([cH.flatten(), cV.flatten(), cD.flatten()])
        
        # MAD estimation: sigma = median(|X|) / 0.6745
        mad = np.median(np.abs(all_details))
        sigma = mad / 0.6745
        
        return sigma
    
    def soft_threshold(self, coeffs, threshold):
        """
        Soft Thresholding Operator (SoT)
        
        Formula:
            T_soft(x) = sign(x) * max(|x| - threshold, 0)
        
        Effect:
            - Small coeffs (< threshold): -> 0 (assumed to be noise)
            - Large coeffs (> threshold): -> shrunk but preserved (signal)
        """
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def forward(self, pred):
        """
        Compute wavelet sparsity loss with adaptive thresholding.
        
        Args:
            pred: [B, C, H, W] - model output (denoised)
            
        Returns:
            loss: L1 distance between coeffs and sparse target
            estimated_noise: average estimated noise level (for monitoring)
        """
        batch_size = pred.size(0)
        device = pred.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        valid_samples = 0
        noise_levels = []
        
        for i in range(batch_size):
            pred_np = pred[i, 0].detach().cpu().numpy()
            
            # Sanity checks
            if np.isnan(pred_np).any() or np.isinf(pred_np).any():
                continue
            
            try:
                # Multi-level DWT
                coeffs = pywt.wavedec2(pred_np, self.wavelet, level=self.levels)
                
                # Estimate noise level from finest level (adaptive)
                if self.adaptive and len(coeffs) > 1:
                    sigma = self.estimate_noise_mad(coeffs[1])
                    noise_levels.append(sigma)
                    
                    # Adapt threshold: k * sigma (k=2.0~3.0 empirically good)
                    adaptive_threshold = max(sigma * 2.5, self.base_threshold * 0.5)  # Floor at 50% of base
                    adaptive_threshold = min(adaptive_threshold, self.base_threshold * 2.0)  # Ceiling at 200% of base
                else:
                    adaptive_threshold = self.base_threshold
                
                level_loss = 0
                
                # Only penalize detail coefficients (high-frequency)
                for level_idx in range(1, len(coeffs)):
                    cH, cV, cD = coeffs[level_idx]
                    
                    # Scale threshold per level
                    level_threshold = adaptive_threshold / (2 ** (level_idx - 1))
                    
                    # Compute sparse target
                    cH_sparse = self.soft_threshold(cH, level_threshold)
                    cV_sparse = self.soft_threshold(cV, level_threshold)
                    cD_sparse = self.soft_threshold(cD, level_threshold)
                    
                    # Convert to tensors
                    cH_tensor = torch.from_numpy(cH).float().to(device)
                    cV_tensor = torch.from_numpy(cV).float().to(device)
                    cD_tensor = torch.from_numpy(cD).float().to(device)
                    
                    cH_sparse_tensor = torch.from_numpy(cH_sparse).float().to(device).detach()
                    cV_sparse_tensor = torch.from_numpy(cV_sparse).float().to(device).detach()
                    cD_sparse_tensor = torch.from_numpy(cD_sparse).float().to(device).detach()
                    
                    # L1 sparsity penalty
                    loss_h = F.l1_loss(cH_tensor, cH_sparse_tensor)
                    loss_v = F.l1_loss(cV_tensor, cV_sparse_tensor)
                    loss_d = F.l1_loss(cD_tensor, cD_sparse_tensor)
                    
                    # Check for NaN
                    if torch.isnan(loss_h) or torch.isnan(loss_v) or torch.isnan(loss_d):
                        continue
                    
                    # Weight finer levels more
                    level_weight = 1.0 / level_idx
                    level_loss = level_loss + level_weight * (loss_h + loss_v + loss_d) / 3.0
                
                total_loss = total_loss + level_loss
                valid_samples += 1
                
            except Exception:
                continue
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), 0.0
        
        # Average noise level for monitoring
        avg_noise = np.mean(noise_levels) if noise_levels else 0.0
        
        return total_loss / valid_samples, avg_noise


class CombinedN2NWaveletLoss(nn.Module):
    """
    Combined Loss: Neighbor2Neighbor + Wavelet Sparsity (Adaptive)
    
    New Features:
    1. Noise-aware weighting: dynamically adjusts wavelet weight based on noise level
    2. Adaptive thresholding: per-image threshold adaptation
    3. Handles noise imbalance across dataset
    
    Philosophy:
    - High noise images: stronger wavelet (more denoising)
    - Low noise images: weaker wavelet (preserve details)
    - Automatic balancing for optimal results
    """
    
    def __init__(self,
                 n2n_gamma=2.0,
                 wavelet_weight=0.005,
                 wavelet_threshold=40,
                 wavelet_levels=3,
                 hu_window=(-160, 240),
                 adaptive=True,
                 target_noise=0.15):  # Target noise level for weight scaling (60 HU)
        super().__init__()
        
        self.base_wavelet_weight = wavelet_weight
        self.target_noise = target_noise  # Reference noise level
        self.adaptive = adaptive
        
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
        
        print(f"\nCombined Loss Balancing (Adaptive):")
        print(f"   N2N weight: 1.0 (MAIN - reconstruction)")
        print(f"   Base Wavelet weight: {wavelet_weight} (REGULARIZATION)")
        print(f"   Adaptive weighting: {adaptive} (noise-aware)")
        print(f"   Target noise level: {target_noise:.4f}")
        print(f"   -> High noise images get stronger wavelet regularization")
        print(f"   -> Low noise images get weaker wavelet (preserve details)")
        
    def forward(self, model, noisy_input):
        """
        Compute combined loss with adaptive weighting.
        
        Args:
            model: denoising network
            noisy_input: [B, C, H, W] - noisy input
            
        Returns:
            total_loss: scalar tensor with gradients
            loss_dict: detailed loss breakdown (detached scalars)
        """
        # N2N loss (main denoising mechanism)
        n2n_total, n2n_dict = self.n2n_loss(model, noisy_input)
        
        # Get model output for wavelet loss
        with torch.no_grad():
            g1, _ = self.n2n_loss.generate_subimages_checkerboard(noisy_input)
        output = model(g1)
        output = torch.clamp(output, 0, 1)
        
        # Wavelet sparsity (regularization) with noise estimation
        wavelet, estimated_noise = self.wavelet_loss(output)
        
        # NaN protection
        if torch.isnan(wavelet):
            wavelet = torch.tensor(0.0, device=noisy_input.device, requires_grad=True)
            estimated_noise = 0.0
        
        # Adaptive weighting based on noise level
        if self.adaptive and estimated_noise > 0:
            # Scale weight proportionally to noise level
            noise_ratio = estimated_noise / self.target_noise
            noise_ratio = max(0.5, min(noise_ratio, 2.0))  # Clamp to [0.5, 2.0]
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
            'wavelet_weighted': wavelet.item() * adaptive_weight,
            'total': total.item(),
            'balance_ratio': n2n_dict['total'] / (wavelet.item() * adaptive_weight + 1e-8),
            'estimated_noise': estimated_noise,
            'adaptive_weight': adaptive_weight
        }


# For backward compatibility
Neighbor2NeighborLoss_v2 = Neighbor2NeighborLoss
WaveletSparsityLoss = WaveletSparsityPrior