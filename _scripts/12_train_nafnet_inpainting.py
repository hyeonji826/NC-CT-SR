#!/usr/bin/env python3
"""
NAFNet Inpainting-based Training Script
노이즈 감지 → 주변 픽셀로 채우기 (Blur 없는 Denoising)
Edge 강화 (0.70) + Detail preservation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd
import time
import json
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models


# ============================================================
# Perceptual Loss (VGG-based for edge/structure preservation)
# ============================================================
class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for structure preservation"""
    def __init__(self, device):
        super().__init__()
        # Use VGG16 features
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        
        # Freeze VGG
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Use specific layers for feature extraction
        self.layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[:9],   # relu2_2
            vgg[:16],  # relu3_3
        ])
        
        self.weights = [1.0, 0.75, 0.5]  # Layer weights
        
    def forward(self, pred, target):
        loss = 0.0
        
        for layer, weight in zip(self.layers, self.weights):
            pred_feat = layer(pred)
            target_feat = layer(target)
            loss += weight * F.mse_loss(pred_feat, target_feat)
        
        return loss


class EdgeLoss(nn.Module):
    """강화된 Sobel 기반 edge loss - 경계 선명도 극대화"""
    def __init__(self):
        super().__init__()
        # Sobel kernels for grayscale
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    dtype=torch.float32).view(1, 1, 3, 3)
        
        # Laplacian for second-order edge detection
        self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], 
                                     dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        # Convert RGB to grayscale (take first channel since all 3 are same for CT)
        pred_gray = pred[:, :1, :, :]  # [B, 1, H, W]
        target_gray = target[:, :1, :, :]  # [B, 1, H, W]
        
        # Move kernels to same device as input
        sobel_x = self.sobel_x.to(pred.device)
        sobel_y = self.sobel_y.to(pred.device)
        laplacian = self.laplacian.to(pred.device)
        
        # Sobel edges (first-order gradient)
        pred_edge_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred_gray, sobel_y, padding=1)
        pred_edge_sobel = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-6)
        
        target_edge_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_edge_y = F.conv2d(target_gray, sobel_y, padding=1)
        target_edge_sobel = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-6)
        
        # Laplacian edges (second-order, sharper)
        pred_edge_lap = torch.abs(F.conv2d(pred_gray, laplacian, padding=1))
        target_edge_lap = torch.abs(F.conv2d(target_gray, laplacian, padding=1))
        
        # Combined loss: Sobel (70%) + Laplacian (30%)
        loss_sobel = F.l1_loss(pred_edge_sobel, target_edge_sobel)
        loss_lap = F.l1_loss(pred_edge_lap, target_edge_lap)
        
        return 0.7 * loss_sobel + 0.3 * loss_lap


class TotalVariationLoss(nn.Module):
    """Total Variation Loss - 과도한 blur 방지"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred):
        # TV loss encourages smoothness but penalizes over-smoothing
        # Take first channel for grayscale
        pred_gray = pred[:, :1, :, :]
        
        # Compute differences
        diff_h = torch.abs(pred_gray[:, :, 1:, :] - pred_gray[:, :, :-1, :])
        diff_w = torch.abs(pred_gray[:, :, :, 1:] - pred_gray[:, :, :, :-1])
        
        # We want to MAXIMIZE variation at edges (inverse TV)
        # So we return negative mean (encourage high variation)
        return -torch.mean(diff_h) - torch.mean(diff_w)


class NAFBlock(nn.Module):
    """NAFNet의 기본 블록"""
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma


class SimpleGate(nn.Module):
    """Simple Gate activation"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """2D Layer Normalization"""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class NAFNet(nn.Module):
    """NAFNet 모델 (Grid Artifact 방지를 위한 Bilinear Upsample 사용)"""
    def __init__(self, img_channel=3, width=32, middle_blk_num=12, 
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        
        for num in dec_blk_nums:
            # PixelShuffle 대신 Bilinear Upsample + Conv 사용
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chan, chan // 2, 3, 1, 1, bias=True)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = self.intro(inp)
        
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x)
        x = x + inp
        
        return x[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class Noise2NoiseDataset(Dataset):
    """Noise2Noise CSV로부터 self-supervised denoising 데이터셋 생성"""
    def __init__(self, csv_path, patch_size=256, augment=True, use_inpainting=True):
        self.csv_path = Path(csv_path)
        self.pairs_df = pd.read_csv(csv_path)
        self.patch_size = patch_size
        self.augment = augment
        self.use_inpainting = use_inpainting
        
        print(f"Loaded {len(self.pairs_df)} Noise2Noise pairs from {csv_path}")
        if use_inpainting:
            print("  Using inpainting-based noise removal (preserve details)")
    
    def detect_noise_pixels(self, slice_2d, threshold=0.15):
        """노이즈 픽셀 감지 (이상치 검출)"""
        # Median filter로 부드러운 버전 생성
        from scipy.ndimage import median_filter
        smoothed = median_filter(slice_2d, size=3)
        
        # 차이가 큰 픽셀 = 노이즈
        diff = np.abs(slice_2d - smoothed)
        noise_mask = diff > threshold
        
        return noise_mask
    
    def inpaint_noise(self, slice_2d, noise_mask):
        """노이즈 부분을 주변 픽셀로 채우기"""
        import cv2
        
        # OpenCV inpainting
        img_uint8 = (slice_2d * 255).astype(np.uint8)
        mask_uint8 = (noise_mask * 255).astype(np.uint8)
        
        # Telea algorithm (fast, detail-preserving)
        inpainted = cv2.inpaint(img_uint8, mask_uint8, inpaintRadius=3, 
                               flags=cv2.INPAINT_TELEA)
        
        return inpainted.astype(np.float32) / 255.0
    
    def __len__(self):
        return len(self.pairs_df)
    
    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        
        # Load NC volume
        import SimpleITK as sitk
        nc_img = sitk.ReadImage(row['nc_path'])
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        # Get two noisy slices
        slice1 = nc_arr[int(row['slice1_idx'])]
        slice2 = nc_arr[int(row['slice2_idx'])]
        
        # Resize if needed
        if slice2.shape != slice1.shape:
            from skimage.transform import resize
            slice2 = resize(slice2, slice1.shape, 
                          order=1, preserve_range=True, anti_aliasing=True)
        
        # Optional: Inpainting-based preprocessing
        if self.use_inpainting:
            # Detect and inpaint noise in slice1
            noise_mask1 = self.detect_noise_pixels(slice1)
            if noise_mask1.any():
                slice1 = self.inpaint_noise(slice1, noise_mask1)
            
            # Detect and inpaint noise in slice2
            noise_mask2 = self.detect_noise_pixels(slice2)
            if noise_mask2.any():
                slice2 = self.inpaint_noise(slice2, noise_mask2)
        
        # Random crop to patch_size
        H, W = slice1.shape
        if H > self.patch_size and W > self.patch_size:
            top = np.random.randint(0, H - self.patch_size)
            left = np.random.randint(0, W - self.patch_size)
            slice1 = slice1[top:top+self.patch_size, left:left+self.patch_size]
            slice2 = slice2[top:top+self.patch_size, left:left+self.patch_size]
        
        # Augmentation
        if self.augment:
            # Random flip
            if np.random.rand() > 0.5:
                slice1 = np.fliplr(slice1)
                slice2 = np.fliplr(slice2)
            if np.random.rand() > 0.5:
                slice1 = np.flipud(slice1)
                slice2 = np.flipud(slice2)
            # Random rotation (90, 180, 270)
            k = np.random.randint(0, 4)
            slice1 = np.rot90(slice1, k)
            slice2 = np.rot90(slice2, k)
        
        # Convert to tensor (add channel dimension)
        slice1 = torch.from_numpy(slice1.copy()).unsqueeze(0)  # [1, H, W]
        slice2 = torch.from_numpy(slice2.copy()).unsqueeze(0)  # [1, H, W]
        
        # Convert grayscale to RGB (repeat channel)
        slice1 = slice1.repeat(3, 1, 1)
        slice2 = slice2.repeat(3, 1, 1)
        
        return slice1, slice2


def calculate_psnr(img1, img2):
    """PSNR 계산"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def load_pretrained_model(model, pretrained_path, device):
    """사전학습 모델 로드 - 유연한 키 매칭"""
    print(f"\n{'='*80}")
    print(f"Loading pretrained model from: {pretrained_path}")
    print(f"{'='*80}")
    
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # 체크포인트 구조 확인
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 모델 구조 확인
    model_dict = model.state_dict()
    pretrained_dict = {}
    
    print("\nAttempting flexible key matching...")
    
    # 전략 1: 직접 매칭
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')  # 'module.' 제거
        
        if new_k in model_dict and model_dict[new_k].shape == v.shape:
            pretrained_dict[new_k] = v
    
    # 전략 2: 부분 매칭 (intro, ending, encoder, decoder blocks)
    if len(pretrained_dict) < len(model_dict) * 0.5:  # 50% 미만이면 부분 매칭 시도
        print("Direct matching insufficient, trying partial matching...")
        
        for k_pre, v_pre in state_dict.items():
            k_pre_clean = k_pre.replace('module.', '')
            
            # Find matching keys by name pattern
            for k_model in model_dict.keys():
                # Check if key patterns match (ignore exact prefix)
                if k_pre_clean.split('.')[-2:] == k_model.split('.')[-2:]:  # Match last 2 components
                    if model_dict[k_model].shape == v_pre.shape:
                        if k_model not in pretrained_dict:  # Don't override
                            pretrained_dict[k_model] = v_pre
    
    # 전략 3: Conv/Linear layer만 선택적으로 로드
    if len(pretrained_dict) < len(model_dict) * 0.3:  # 30% 미만이면 Conv/Linear만
        print("Partial matching insufficient, loading Conv/Linear layers only...")
        
        for k_model, v_model in model_dict.items():
            # Look for similar conv/linear layers
            layer_type = k_model.split('.')[-1]  # weight or bias
            
            for k_pre, v_pre in state_dict.items():
                k_pre_clean = k_pre.replace('module.', '')
                
                # Match by layer type and shape
                if k_pre_clean.endswith(layer_type) and v_pre.shape == v_model.shape:
                    # Check if it's a conv or linear layer
                    if 'conv' in k_pre_clean.lower() or 'linear' in k_pre_clean.lower():
                        if k_model not in pretrained_dict:
                            pretrained_dict[k_model] = v_pre
                            break
    
    # 로드된 파라미터 통계
    loaded_keys = len(pretrained_dict)
    total_keys = len(model_dict)
    
    print(f"\n{'='*80}")
    print(f"Loading Results:")
    print(f"  Total parameters in model: {total_keys}")
    print(f"  Loaded from pretrained: {loaded_keys}")
    print(f"  Loading ratio: {loaded_keys/total_keys*100:.1f}%")
    
    # 로드된 레이어 타입 분석
    loaded_types = {}
    for k in pretrained_dict.keys():
        layer_name = k.split('.')[0]  # First component (intro, encoders, etc.)
        loaded_types[layer_name] = loaded_types.get(layer_name, 0) + 1
    
    print(f"\n  Loaded by module:")
    for layer_name, count in sorted(loaded_types.items()):
        print(f"    {layer_name}: {count} parameters")
    
    if loaded_keys == 0:
        print(f"\n⚠️  WARNING: No parameters loaded!")
        print(f"  Model architecture may be incompatible.")
        print(f"  Training from scratch...")
        return model
    elif loaded_keys < total_keys * 0.5:
        print(f"\n⚠️  WARNING: Only {loaded_keys/total_keys*100:.1f}% parameters loaded!")
        print(f"  Some layers will be randomly initialized.")
    else:
        print(f"\n✓ Successfully loaded {loaded_keys/total_keys*100:.1f}% of parameters!")
    
    # 파라미터 로드
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f"{'='*80}\n")
    
    return model


def train_epoch(model, dataloader, criterion_dict, optimizer, device, epoch, args):
    """1 epoch 학습 with Multi-task Loss"""
    model.train()
    total_loss = 0
    total_mse_loss = 0
    total_edge_loss = 0
    total_tv_loss = 0
    total_psnr = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for noisy1, noisy2 in pbar:
        noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
        
        # Forward
        denoised = model(noisy1)
        
        # Multi-task loss
        loss_mse = criterion_dict['mse'](denoised, noisy2)
        
        # Edge loss (if enabled)
        if args.use_edge:
            loss_edge = criterion_dict['edge'](denoised, noisy2)
        else:
            loss_edge = torch.tensor(0.0, device=device)
        
        # TV loss (if enabled and after warmup)
        if args.use_tv and epoch >= args.tv_warmup:
            loss_tv = criterion_dict['tv'](denoised)
        else:
            loss_tv = torch.tensor(0.0, device=device)
        
        # Combined loss with adjustable weights
        loss = (args.mse_weight * loss_mse + 
                args.edge_weight * loss_edge +
                args.tv_weight * loss_tv)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            psnr = calculate_psnr(denoised, noisy2)
        
        total_loss += loss.item()
        total_mse_loss += loss_mse.item()
        total_edge_loss += loss_edge.item() if isinstance(loss_edge, torch.Tensor) else 0
        total_tv_loss += loss_tv.item() if isinstance(loss_tv, torch.Tensor) else 0
        total_psnr += psnr.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mse': f'{loss_mse.item():.4f}',
            'edge': f'{loss_edge.item():.4f}' if isinstance(loss_edge, torch.Tensor) else '0',
            'psnr': f'{psnr.item():.2f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse_loss / len(dataloader)
    avg_edge = total_edge_loss / len(dataloader)
    avg_tv = total_tv_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_mse, avg_edge, avg_tv, avg_psnr


def validate(model, dataloader, criterion_dict, device, args, epoch=None, sample_dir=None):
    """Validation (Noise2Noise with Multi-task Loss)"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    
    # Save samples
    save_samples = (epoch is not None and sample_dir is not None and epoch % 5 == 0)
    saved_count = 0
    max_samples = 3
    
    with torch.no_grad():
        for batch_idx, (noisy1, noisy2) in enumerate(tqdm(dataloader, desc="Validating")):
            noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
            
            denoised = model(noisy1)
            loss = criterion_dict['mse'](denoised, noisy2)  # Use MSE for validation
            psnr = calculate_psnr(denoised, noisy2)
            
            total_loss += loss.item()
            total_psnr += psnr.item()
            
            # Save sample images
            if save_samples and saved_count < max_samples and batch_idx == 0:
                for i in range(min(max_samples - saved_count, noisy1.size(0))):
                    # Convert to numpy and save
                    noisy1_img = noisy1[i].cpu().numpy()[0]  # Take first channel
                    noisy2_img = noisy2[i].cpu().numpy()[0]
                    denoised_img = denoised[i].cpu().numpy()[0]
                    
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(noisy1_img, cmap='gray', vmin=0, vmax=1)
                    axes[0].set_title('Input (Noisy 1)', fontsize=12, fontweight='bold')
                    axes[0].axis('off')
                    
                    axes[1].imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
                    axes[1].set_title('Denoised', fontsize=12, fontweight='bold')
                    axes[1].axis('off')
                    
                    axes[2].imshow(noisy2_img, cmap='gray', vmin=0, vmax=1)
                    axes[2].set_title('Target (Noisy 2)', fontsize=12, fontweight='bold')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    save_path = sample_dir / f'best_epoch_{epoch:03d}.png'
                    plt.savefig(save_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    saved_count += 1
                    if saved_count >= max_samples:
                        break
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_psnr


def save_best_samples(model, dataloader, device, sample_dir, epoch, psnr, metrics_dict):
    """Best 모델 저장 시 샘플 이미지 + 지표 생성"""
    model.eval()
    
    with torch.no_grad():
        # 첫 번째 배치만 사용
        for noisy1, noisy2 in dataloader:
            noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
            denoised = model(noisy1)
            
            # 최대 3개 샘플
            num_samples = min(3, noisy1.size(0))
            
            import matplotlib.pyplot as plt
            for i in range(num_samples):
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Convert to numpy
                noisy1_img = noisy1[i].cpu().numpy()[0]
                denoised_img = denoised[i].cpu().numpy()[0]
                noisy2_img = noisy2[i].cpu().numpy()[0]
                
                # Row 1: Images
                axes[0, 0].imshow(noisy1_img, cmap='gray', vmin=0, vmax=1)
                axes[0, 0].set_title('Input (Noisy 1)', fontsize=14, fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(denoised_img, cmap='gray', vmin=0, vmax=1)
                axes[0, 1].set_title('Denoised (Output)', fontsize=14, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(noisy2_img, cmap='gray', vmin=0, vmax=1)
                axes[0, 2].set_title('Target (Noisy 2)', fontsize=14, fontweight='bold')
                axes[0, 2].axis('off')
                
                # Row 2: Difference maps & metrics
                diff_input = np.abs(noisy1_img - noisy2_img)
                diff_output = np.abs(denoised_img - noisy2_img)
                improvement = diff_input - diff_output
                
                im1 = axes[1, 0].imshow(diff_input, cmap='hot', vmin=0, vmax=0.3)
                axes[1, 0].set_title('Input vs Target\n(Noise Level)', fontsize=12)
                axes[1, 0].axis('off')
                plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
                
                im2 = axes[1, 1].imshow(diff_output, cmap='hot', vmin=0, vmax=0.3)
                axes[1, 1].set_title('Denoised vs Target\n(Residual)', fontsize=12)
                axes[1, 1].axis('off')
                plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
                
                im3 = axes[1, 2].imshow(improvement, cmap='RdYlGn', vmin=-0.1, vmax=0.1)
                axes[1, 2].set_title('Improvement\n(Green=Better)', fontsize=12)
                axes[1, 2].axis('off')
                plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
                
                # Add metrics text
                metrics_text = f"""
Best Model Metrics (Epoch {epoch})
━━━━━━━━━━━━━━━━━━━━━━
PSNR: {psnr:.2f} dB
Train Loss: {metrics_dict.get('train_loss', 0):.4f}
Val Loss: {metrics_dict.get('val_loss', 0):.4f}
MSE: {metrics_dict.get('train_mse', 0):.4f}
Edge: {metrics_dict.get('train_edge', 0):.4f}
TV: {metrics_dict.get('train_tv', 0):.4f}
                """
                
                fig.text(0.02, 0.98, metrics_text, fontsize=11, 
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.suptitle(f'Best Model - Epoch {epoch} - PSNR: {psnr:.2f} dB', 
                           fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0, 1, 0.96])
                
                save_path = sample_dir / f'best_epoch_{epoch:03d}_sample_{i+1}.png'
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"     Saved {num_samples} sample images with metrics")
            break
    
    model.train()


def load_checkpoint_if_exists(checkpoint_dir, model, optimizer, scheduler):
    """체크포인트가 있으면 로드하여 이어서 학습"""
    best_checkpoint = checkpoint_dir / 'best_model.pth'
    
    if best_checkpoint.exists():
        print(f"\n{'='*80}")
        print(f"Found existing checkpoint: {best_checkpoint}")
        response = input("Resume training from this checkpoint? (y/n): ")
        
        if response.lower() == 'y':
            print("Loading checkpoint...")
            checkpoint = torch.load(best_checkpoint, map_location=model.device if hasattr(model, 'device') else 'cpu')
            
            model.load_state_dict(checkpoint['model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_psnr = checkpoint.get('best_psnr', 0)
            
            print(f"✓ Resumed from epoch {start_epoch-1}")
            print(f"  Best PSNR so far: {best_psnr:.2f} dB")
            print(f"{'='*80}\n")
            
            return start_epoch, best_psnr
        else:
            print("Starting fresh training...")
            print(f"{'='*80}\n")
    
    return 1, 0  # start_epoch=1, best_psnr=0


def train(args):
    """메인 학습 함수"""
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 디렉토리 생성
    output_dir = Path(args.output_dir)
    exp_name = args.exp_name if args.exp_name else 'nafnet_nc2ce'
    
    # experiments 폴더 안에 저장
    exp_dir = output_dir / 'experiments' / exp_name
    checkpoint_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    sample_dir = exp_dir / 'samples'
    
    # 기존 폴더가 있어도 덮어쓰기 (exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExperiment directory: {exp_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Logs: {log_dir}")
    print(f"  - Samples: {sample_dir}")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    
    # 데이터셋 생성
    print("\nLoading datasets...")
    train_dataset = Noise2NoiseDataset(
        args.train_csv,
        patch_size=args.patch_size,
        augment=True,
        use_inpainting=args.use_inpainting
    )
    
    # Validation dataset (if provided)
    val_dataloader = None
    if args.val_csv:
        val_dataset = Noise2NoiseDataset(
            args.val_csv,
            patch_size=args.patch_size,
            augment=False,
            use_inpainting=args.use_inpainting
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 모델 생성
    print("\nCreating model...")
    model = NAFNet(
        img_channel=3,
        width=args.width,
        middle_blk_num=args.middle_blk_num,
        enc_blk_nums=args.enc_blk_nums,
        dec_blk_nums=args.dec_blk_nums
    ).to(device)
    
    # 사전학습 모델 로드
    if args.pretrained:
        model = load_pretrained_model(model, args.pretrained, device)
    
    # Loss functions dictionary
    criterion_dict = {
        'mse': nn.L1Loss()  # L1 for main denoising
    }
    
    # Add edge loss (강화된 버전)
    if args.use_edge:
        print("Initializing Enhanced Edge Loss (Sobel + Laplacian)...")
        criterion_dict['edge'] = EdgeLoss()
    
    # Add TV loss for sharpness
    if args.use_tv:
        print("Initializing Total Variation Loss (Anti-blur)...")
        criterion_dict['tv'] = TotalVariationLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Check for existing checkpoint and resume if requested
    start_epoch, best_psnr = load_checkpoint_if_exists(checkpoint_dir, model, optimizer, scheduler)
    
    print(f"\n{'='*80}")
    print("Inpainting-based Training: NAFNet-SIDD + Noise Inpainting + Sharp Edges")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_dataset)}")
    if val_dataloader:
        print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs} (starting from epoch {start_epoch})")
    print(f"Learning rate: {args.lr}")
    print(f"\nPreprocessing:")
    print(f"  Inpainting: {args.use_inpainting} (노이즈 감지 → 주변 픽셀로 채우기)")
    print(f"\nLoss Configuration:")
    print(f"  MSE weight: {args.mse_weight} (기본 denoising)")
    if args.use_edge:
        print(f"  Edge weight: {args.edge_weight} (경계 선명도 극대화 - Sobel+Laplacian)")
    if args.use_tv:
        print(f"  TV weight: {args.tv_weight} (Anti-blur, warmup: {args.tv_warmup} epochs)")
    print(f"Strategy: Detail-preserving denoising with sharp edges")
    if start_epoch > 1:
        print(f"\n✓ Resuming from epoch {start_epoch}, Best PSNR: {best_psnr:.2f} dB")
    print(f"{'='*80}\n")
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_mse, train_edge, train_tv, train_psnr = train_epoch(
            model, train_dataloader, criterion_dict, optimizer, device, epoch, args
        )
        
        # Validate
        if val_dataloader:
            val_loss, val_psnr = validate(model, val_dataloader, criterion_dict, device, args, epoch, sample_dir)
        else:
            val_loss, val_psnr = 0, 0
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{args.epochs} - {epoch_time:.1f}s")
        print(f"  Train - Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Edge: {train_edge:.4f}, TV: {train_tv:.4f}), PSNR: {train_psnr:.2f} dB")
        if val_dataloader:
            print(f"  Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/train_mse', train_mse, epoch)
        writer.add_scalar('Loss/train_edge', train_edge, epoch)
        writer.add_scalar('Loss/train_tv', train_tv, epoch)
        writer.add_scalar('PSNR/train', train_psnr, epoch)
        if val_dataloader:
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('PSNR/val', val_psnr, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_psnr': train_psnr,
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'config': {
                    'width': args.width,
                    'middle_blk_num': args.middle_blk_num,
                    'enc_blk_nums': args.enc_blk_nums,
                    'dec_blk_nums': args.dec_blk_nums
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model (keep multiple checkpoints)
        current_psnr = val_psnr if val_dataloader else train_psnr
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            
            # Prepare metrics dictionary
            metrics_dict = {
                'train_loss': train_loss,
                'train_mse': train_mse,
                'train_edge': train_edge,
                'train_tv': train_tv,
                'val_loss': val_loss,
                'val_psnr': val_psnr
            }
            
            # Save with epoch number to keep history
            best_path = checkpoint_dir / f'best_model_epoch_{epoch}_psnr_{best_psnr:.2f}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'train_loss': train_loss,
                'train_psnr': train_psnr,
                'val_loss': val_loss,
                'val_psnr': val_psnr,
                'metrics': metrics_dict,
                'config': {
                    'width': args.width,
                    'middle_blk_num': args.middle_blk_num,
                    'enc_blk_nums': args.enc_blk_nums,
                    'dec_blk_nums': args.dec_blk_nums
                }
            }, best_path)
            
            # Also save as 'best_model.pth' for easy access (latest best)
            latest_best = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
                'metrics': metrics_dict,
                'config': {
                    'width': args.width,
                    'middle_blk_num': args.middle_blk_num,
                    'enc_blk_nums': args.enc_blk_nums,
                    'dec_blk_nums': args.dec_blk_nums
                }
            }, latest_best)
            
            # Save sample images with metrics
            save_best_samples(model, val_dataloader if val_dataloader else train_dataloader, 
                            device, sample_dir, epoch, best_psnr, metrics_dict)
            
            print(f"  ⭐ New best model! PSNR: {best_psnr:.2f} dB (Epoch {epoch})")
            print(f"     Saved: {best_path.name}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        print()
    
    writer.close()
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Results saved to: {exp_dir}")
    print(f"  - Best model: {checkpoint_dir / 'best_model.pth'}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Samples: {sample_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='NAFNet Inpainting-based Training (Detail-preserving Denoising + Sharp Edges)')
    
    # Data
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training Noise2Noise pairs CSV')
    parser.add_argument('--val_csv', type=str, default=None,
                       help='Path to validation Noise2Noise pairs CSV (optional)')
    parser.add_argument('--output_dir', type=str, default='Outputs',
                       help='Output directory for experiments')
    parser.add_argument('--exp_name', type=str, default='nafnet_inpainting_sharp',
                       help='Experiment name (creates subdirectory in output_dir)')
    
    # Preprocessing
    parser.add_argument('--use_inpainting', action='store_true', default=True,
                       help='Use inpainting-based noise removal (detail-preserving)')
    
    # Pretrained model
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Path to pretrained model (e.g., NAFSSR-B_4x.pth)')
    
    # Model architecture
    parser.add_argument('--width', type=int, default=32,
                       help='Base channel width')
    parser.add_argument('--middle_blk_num', type=int, default=12,
                       help='Number of middle blocks')
    parser.add_argument('--enc_blk_nums', type=int, nargs='+', default=[2, 2, 4, 8],
                       help='Number of encoder blocks per stage')
    parser.add_argument('--dec_blk_nums', type=int, nargs='+', default=[2, 2, 2, 2],
                       help='Number of decoder blocks per stage')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--patch_size', type=int, default=256,
                       help='Training patch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (낮게 설정 for fine-tuning)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--mse_weight', type=float, default=0.5,
                       help='Weight for MSE loss (pixel-level)')
    parser.add_argument('--use_edge', action='store_true', default=True,
                       help='Use enhanced edge loss (Sobel + Laplacian)')
    parser.add_argument('--edge_weight', type=float, default=0.4,
                       help='Weight for edge loss (sharpness) - 높일수록 경계 선명')
    parser.add_argument('--use_tv', action='store_true', default=True,
                       help='Use TV loss for anti-blur')
    parser.add_argument('--tv_weight', type=float, default=0.1,
                       help='Weight for TV loss (anti-blur)')
    parser.add_argument('--tv_warmup', type=int, default=3,
                       help='Epochs to wait before enabling TV loss')
    
    # Checkpointing
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()