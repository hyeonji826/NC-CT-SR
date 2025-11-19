import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

class WaveletLoss(nn.Module):
    """
    논문 기반 Wavelet Loss with Soft Thresholding
    Reference: "복부 CT 영상의 화질 개선 방법에 대한 연구" (2023)
    
    핵심 개선:
    1. Soft Thresholding 적용 (논문의 SoT 방식)
    2. Multi-level decomposition (2-level)
    3. Adaptive threshold per level
    """
    def __init__(self, wavelet='haar', threshold=50, levels=2, normalize_threshold=True):
        super().__init__()
        self.wavelet = wavelet
        self.threshold = threshold  # 논문에서 제안한 최적값: 50
        self.levels = levels
        self.normalize_threshold = normalize_threshold
        
        print(f"📊 WaveletLoss Initialized:")
        print(f"   Wavelet: {wavelet}")
        print(f"   Threshold: {threshold}")
        print(f"   Levels: {levels}")
    
    def soft_threshold(self, coeffs, threshold):
        """
        Soft Thresholding (SoT) - 논문의 수식 (2)
        
        w'(i) = 0           if |w(i)| < T
              = w(i) - T   if w(i) ≥ T  
              = w(i) + T   if w(i) ≤ -T
        
        효과: 임계값보다 작은 계수(노이즈)는 0으로, 큰 계수(신호)는 수축
        """
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)
    
    def forward(self, pred, target):
        """
        pred, target: [B, 1, H, W] - normalized to [0, 1]
        """
        batch_size = pred.size(0)
        device = pred.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        valid_samples = 0
        
        for i in range(batch_size):
            # Move to CPU for pywt
            pred_np = pred[i, 0].detach().cpu().numpy()
            target_np = target[i, 0].detach().cpu().numpy()
            
            # Sanity checks
            if np.isnan(pred_np).any() or np.isnan(target_np).any():
                continue
            if np.isinf(pred_np).any() or np.isinf(target_np).any():
                continue
            
            try:
                # Multi-level DWT decomposition
                coeffs_pred = pywt.wavedec2(pred_np, self.wavelet, level=self.levels)
                coeffs_target = pywt.wavedec2(target_np, self.wavelet, level=self.levels)
                
                # coeffs = [cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1)]
                
                level_loss = 0
                
                for level_idx in range(1, len(coeffs_pred)):  # Skip approximation (cA)
                    cH_pred, cV_pred, cD_pred = coeffs_pred[level_idx]
                    cH_target, cV_target, cD_target = coeffs_target[level_idx]
                    
                    # Adaptive threshold per level
                    # Higher levels (coarser) get lower threshold
                    level_threshold = self.threshold / (2 ** (level_idx - 1))
                    
                    # Normalize threshold to match [0, 1] range if needed
                    if self.normalize_threshold:
                        # Input is [0, 1], so scale threshold accordingly
                        level_threshold = level_threshold / 255.0
                    
                    # Apply Soft Thresholding to high-frequency components
                    cH_pred_t = self.soft_threshold(cH_pred, level_threshold)
                    cV_pred_t = self.soft_threshold(cV_pred, level_threshold)
                    cD_pred_t = self.soft_threshold(cD_pred, level_threshold)
                    
                    cH_target_t = self.soft_threshold(cH_target, level_threshold)
                    cV_target_t = self.soft_threshold(cV_target, level_threshold)
                    cD_target_t = self.soft_threshold(cD_target, level_threshold)
                    
                    # Convert to tensors
                    cH_pred_tensor = torch.from_numpy(cH_pred_t).float().to(device)
                    cH_target_tensor = torch.from_numpy(cH_target_t).float().to(device)
                    
                    cV_pred_tensor = torch.from_numpy(cV_pred_t).float().to(device)
                    cV_target_tensor = torch.from_numpy(cV_target_t).float().to(device)
                    
                    cD_pred_tensor = torch.from_numpy(cD_pred_t).float().to(device)
                    cD_target_tensor = torch.from_numpy(cD_target_t).float().to(device)
                    
                    # Compute L1 loss on thresholded coefficients
                    loss_h = F.l1_loss(cH_pred_tensor, cH_target_tensor)
                    loss_v = F.l1_loss(cV_pred_tensor, cV_target_tensor)
                    loss_d = F.l1_loss(cD_pred_tensor, cD_target_tensor)
                    
                    # Check for NaN
                    if torch.isnan(loss_h) or torch.isnan(loss_v) or torch.isnan(loss_d):
                        continue
                    
                    # Weight by level (finer levels get higher weight)
                    level_weight = 1.0 / level_idx
                    level_loss = level_loss + level_weight * (loss_h + loss_v + loss_d) / 3.0
                
                total_loss = total_loss + level_loss
                valid_samples += 1
                
            except Exception as e:
                print(f"⚠️ Wavelet error in sample {i}: {e}")
                continue
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return total_loss / valid_samples


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
    """
    L1 + SSIM + Wavelet (with Soft Thresholding)
    
    논문 기반 개선:
    - Wavelet Loss에 Soft Thresholding 적용
    - Multi-level DWT로 다양한 주파수 대역 처리
    - Adaptive threshold로 레벨별 최적화
    """
    def __init__(self, l1_weight=1.0, ssim_weight=0.5, wavelet_weight=0.1, 
                 wavelet_threshold=50):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.wavelet_weight = wavelet_weight
        
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.wavelet_loss = WaveletLoss(
            wavelet='haar',
            threshold=wavelet_threshold,  # 논문의 최적값: 50
            levels=2,
            normalize_threshold=True
        )
        
        print(f"\n📊 CombinedLoss Configuration:")
        print(f"   L1 weight: {l1_weight}")
        print(f"   SSIM weight: {ssim_weight}")
        print(f"   Wavelet weight: {wavelet_weight}")
        print(f"   Wavelet threshold: {wavelet_threshold}")
    
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
        
        # L1 loss (always active)
        l1 = self.l1_loss(pred, target)
        
        # SSIM loss
        if self.ssim_weight > 0:
            ssim = self.ssim_loss(pred, target)
        else:
            ssim = torch.tensor(0.0, device=pred.device)
        
        # Wavelet loss with Soft Thresholding
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