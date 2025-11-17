# E:\LD-CT SR\_scripts_4_wavelet\losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class WaveletLoss(nn.Module):
    """Wavelet-based high-frequency loss (Fixed for stability)"""
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
    
    def forward(self, pred, target):
        """
        pred, target: [B, 1, H, W]
        """
        batch_size = pred.size(0)
        device = pred.device
        total_loss = 0
        
        for i in range(batch_size):
            # Move to CPU for wavelet (PyWavelets doesn't support GPU)
            pred_np = pred[i, 0].detach().cpu().numpy()
            target_np = target[i, 0].detach().cpu().numpy()
            
            # Check for nan/inf
            if np.isnan(pred_np).any() or np.isnan(target_np).any():
                print("⚠️ Warning: NaN detected in wavelet input")
                continue
            
            if np.isinf(pred_np).any() or np.isinf(target_np).any():
                print("⚠️ Warning: Inf detected in wavelet input")
                continue
            
            try:
                # 2D Wavelet decomposition
                coeffs_pred = pywt.dwt2(pred_np, self.wavelet)
                coeffs_target = pywt.dwt2(target_np, self.wavelet)
                
                # LL, (LH, HL, HH)
                cA_pred, (cH_pred, cV_pred, cD_pred) = coeffs_pred
                cA_target, (cH_target, cV_target, cD_target) = coeffs_target
                
                # Convert to tensors
                cH_pred_t = torch.from_numpy(cH_pred).float().to(device)
                cH_target_t = torch.from_numpy(cH_target).float().to(device)
                cV_pred_t = torch.from_numpy(cV_pred).float().to(device)
                cV_target_t = torch.from_numpy(cV_target).float().to(device)
                cD_pred_t = torch.from_numpy(cD_pred).float().to(device)
                cD_target_t = torch.from_numpy(cD_target).float().to(device)
                
                # High-frequency loss (LH + HL + HH)
                loss_h = F.l1_loss(cH_pred_t, cH_target_t)
                loss_v = F.l1_loss(cV_pred_t, cV_target_t)
                loss_d = F.l1_loss(cD_pred_t, cD_target_t)
                
                # Check for nan
                if torch.isnan(loss_h) or torch.isnan(loss_v) or torch.isnan(loss_d):
                    print("⚠️ Warning: NaN in wavelet loss computation")
                    continue
                
                total_loss += (loss_h + loss_v + loss_d) / 3.0
                
            except Exception as e:
                print(f"⚠️ Wavelet error: {e}")
                continue
        
        if batch_size == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / batch_size


class SSIMLoss(nn.Module):
    """SSIM Loss (Fixed for stability)"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def gaussian_window(self, window_size, sigma=1.5):
        gauss = torch.exp(torch.tensor([
            -((x - window_size//2)**2) / (2.0 * sigma**2) 
            for x in range(window_size)
        ], dtype=torch.float32))
        return gauss / gauss.sum()
    
    def forward(self, pred, target):
        """
        pred, target: [B, 1, H, W]
        """
        # Check for nan/inf
        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("⚠️ Warning: NaN in SSIM input")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        if torch.isinf(pred).any() or torch.isinf(target).any():
            print("⚠️ Warning: Inf in SSIM input")
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Create Gaussian window
        window_1d = self.gaussian_window(self.window_size).unsqueeze(0)
        window = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window.to(pred.device)
        
        mu1 = F.conv2d(pred, window, padding=self.window_size//2)
        mu2 = F.conv2d(target, window, padding=self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size//2) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        ssim_loss = 1 - ssim_map.mean()
        
        # Clamp to prevent extreme values
        ssim_loss = torch.clamp(ssim_loss, 0, 2)
        
        return ssim_loss


class CombinedLoss(nn.Module):
    """L1 + SSIM + Wavelet (with stability checks)"""
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, wavelet_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.wavelet_weight = wavelet_weight
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.wavelet_loss = WaveletLoss()
    
    def forward(self, pred, target):
        # Clamp predictions to [0, 1]
        pred = torch.clamp(pred, 0, 1)
        
        # Check for nan/inf
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("⚠️ Warning: NaN/Inf in prediction!")
            return torch.tensor(float('inf'), device=pred.device), {
                'l1': float('inf'), 'ssim': float('inf'), 
                'wavelet': float('inf'), 'total': float('inf')
            }
        
        # L1 loss
        l1 = self.l1_loss(pred, target)
        
        # SSIM loss (with weight check)
        if self.ssim_weight > 0:
            ssim = self.ssim_loss(pred, target)
        else:
            ssim = torch.tensor(0.0, device=pred.device)
        
        # Wavelet loss (with weight check)
        if self.wavelet_weight > 0:
            wavelet = self.wavelet_loss(pred, target)
        else:
            wavelet = torch.tensor(0.0, device=pred.device)
        
        # Check individual losses
        if torch.isnan(l1):
            l1 = torch.tensor(0.0, device=pred.device, requires_grad=True)
        if torch.isnan(ssim):
            ssim = torch.tensor(0.0, device=pred.device, requires_grad=True)
        if torch.isnan(wavelet):
            wavelet = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        total = (self.l1_weight * l1 + 
                 self.ssim_weight * ssim + 
                 self.wavelet_weight * wavelet)
        
        return total, {
            'l1': l1.item(),
            'ssim': ssim.item(),
            'wavelet': wavelet.item(),
            'total': total.item()
        }