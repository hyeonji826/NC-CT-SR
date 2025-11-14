#!/usr/bin/env python3
"""
05_train_enhancer.py
Stage-2: Clean NC → Enhanced NC (Quality Enhancement)

목표: 표준선량 NC를 더 고화질로!
방법: Self-supervised + Perceptual enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import cv2
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

try:
    from pytorch_msssim import ssim
except ImportError:
    ssim = None

try:
    from torchvision.models import vgg19, VGG19_Weights
except ImportError:
    vgg19 = None


# ============================================================
# NAFNet Model (동일)
# ============================================================
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
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
        
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=12, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
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
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
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
        x = x + inp  # Residual
        
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ============================================================
# Perceptual Loss (VGG-based)
# ============================================================
class PerceptualLoss(nn.Module):
    """VGG19 기반 Perceptual Loss"""
    def __init__(self):
        super().__init__()
        if vgg19 is not None:
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            self.layers = nn.ModuleList([
                vgg[:4],   # relu1_2
                vgg[:9],   # relu2_2
                vgg[:18],  # relu3_4
            ])
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
        else:
            self.layers = None
    
    def forward(self, pred, target):
        if self.layers is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Grayscale to RGB
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)
        
        loss = 0.0
        for layer in self.layers:
            pred_feat = layer(pred_rgb)
            target_feat = layer(target_rgb)
            loss += F.l1_loss(pred_feat, target_feat)
        
        return loss / len(self.layers)


# ============================================================
# Enhanced Loss
# ============================================================
class EnhancedLoss(nn.Module):
    """
    Quality enhancement를 위한 loss
    - L1: 기본 픽셀 복원
    - SSIM: 구조 유사도
    - Edge: 경계 선명도
    - Perceptual: 질감/디테일
    """
    def __init__(self, l1_weight=0.5, ssim_weight=0.2, edge_weight=0.2, perceptual_weight=0.1):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.perceptual_weight = perceptual_weight
        
        # Edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Perceptual loss
        self.perceptual_loss = PerceptualLoss()

    def edge_loss(self, pred, target):
        pred_edge_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edge_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edge_y = F.conv2d(target, self.sobel_y, padding=1)
        
        pred_edge = torch.sqrt(pred_edge_x**2 + pred_edge_y**2 + 1e-8)
        target_edge = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)
        
        return F.l1_loss(pred_edge, target_edge)

    def forward(self, pred, target):
        # L1
        loss_l1 = F.l1_loss(pred, target)
        
        # SSIM
        if ssim is not None:
            loss_ssim = 1 - ssim(pred, target, data_range=1.0, size_average=True)
        else:
            loss_ssim = torch.tensor(0.0, device=pred.device)
        
        # Edge
        loss_edge = self.edge_loss(pred, target)
        
        # Perceptual
        loss_perceptual = self.perceptual_loss(pred, target)
        
        # Combined
        total_loss = (self.l1_weight * loss_l1 + 
                      self.ssim_weight * loss_ssim + 
                      self.edge_weight * loss_edge +
                      self.perceptual_weight * loss_perceptual)
        
        return total_loss, {
            'l1': loss_l1.item(),
            'ssim': loss_ssim.item(),
            'edge': loss_edge.item(),
            'perceptual': loss_perceptual.item(),
            'total': total_loss.item()
        }


# ============================================================
# Dataset with Self-supervised Degradation
# ============================================================
class EnhancementDataset(Dataset):
    """
    Direct quality enhancement
    Clean NC를 input으로, Stage-1 모델이 더 향상시킴
    
    핵심: Stage-1에서 배운 "품질 향상 능력"을 Clean NC에 적용
    """
    def __init__(self, csv_path, image_size=512, augment=True):
        self.pairs = pd.read_csv(csv_path)
        self.image_size = image_size
        self.augment = augment
        
    def __len__(self):
        return len(self.pairs)
    
    def load_slice(self, nifti_path, slice_idx):
        img = sitk.ReadImage(str(nifti_path))
        arr = sitk.GetArrayFromImage(img)[slice_idx]
        return arr
    
    def augment_image(self, img):
        if not self.augment:
            return img
        
        if random.random() > 0.5:
            img = np.fliplr(img)
        
        if random.random() > 0.5:
            img = np.flipud(img)
        
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k)
        
        return img
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        # Load clean NC
        clean = self.load_slice(row['clean_path'], row['slice_idx'])
        
        # Augment
        clean = self.augment_image(clean)
        
        # Resize
        clean = cv2.resize(clean, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        clean = torch.from_numpy(clean).float().unsqueeze(0)
        
        # Input = Clean NC (그대로)
        # Target = Clean NC (그대로, identity mapping 학습)
        # 하지만 모델은 Stage-1 지식으로 더 개선된 결과 출력
        
        return {'input': clean, 'target': clean}


# ============================================================
# Trainer
# ============================================================
class EnhancerTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        
        self.exp_dir = Path(args.exp_dir)
        self.ckpt_dir = self.exp_dir / 'ckpt'
        self.samples_dir = self.exp_dir / 'samples'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Model
        self.model = NAFNet(img_channel=1, width=args.width).to(self.device)
        
        # Load from stage-1 (optional)
        if args.pretrained_denoiser:
            checkpoint = torch.load(args.pretrained_denoiser, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"Loaded denoiser weights from: {args.pretrained_denoiser}")
        
        # Loss
        self.criterion = EnhancedLoss(
            l1_weight=args.l1_weight,
            ssim_weight=args.ssim_weight,
            edge_weight=args.edge_weight,
            perceptual_weight=args.perceptual_weight
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        # Dataset
        self.train_dataset = EnhancementDataset(
            csv_path=args.train_csv,
            image_size=args.image_size,
            augment=True
        )
        
        self.val_dataset = EnhancementDataset(
            csv_path=args.val_csv,
            image_size=args.image_size,
            augment=False
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        if args.resume:
            self.load_checkpoint(args.resume)
    
    def load_checkpoint(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            return
        
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch in pbar:
            inp = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            
            pred = self.model(inp)
            loss, loss_dict = self.criterion(pred, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_losses = []
        psnr_vals = []
        ssim_vals = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            inp = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            
            pred = self.model(inp)
            loss, _ = self.criterion(pred, target)
            
            val_losses.append(loss.item())
            
            for i in range(pred.shape[0]):
                pred_np = pred[i, 0].cpu().numpy()
                target_np = target[i, 0].cpu().numpy()
                psnr_vals.append(psnr_metric(target_np, pred_np, data_range=1.0))
                ssim_vals.append(ssim_metric(target_np, pred_np, data_range=1.0))
        
        return np.mean(val_losses), np.mean(psnr_vals), np.mean(ssim_vals)
    
    @torch.no_grad()
    def save_samples(self, epoch):
        self.model.eval()
        batch = next(iter(self.val_loader))
        inp = batch['input'][:4].to(self.device)
        target = batch['target'][:4].to(self.device)
        
        pred = self.model(inp)
        
        psnr_vals = []
        ssim_vals = []
        for i in range(4):
            pred_np = pred[i, 0].cpu().numpy()
            target_np = target[i, 0].cpu().numpy()
            psnr_vals.append(psnr_metric(target_np, pred_np, data_range=1.0))
            ssim_vals.append(ssim_metric(target_np, pred_np, data_range=1.0))
        
        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'Epoch {epoch+1} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}', fontsize=14)
        
        for i in range(4):
            axes[0, i].imshow(inp[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Input (Clean NC)')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(pred[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Output (Enhanced, PSNR: {psnr_vals[i]:.2f})')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(target[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title(f'Target (Clean NC, same as input)')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.samples_dir / f'epoch_{epoch+1:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        torch.save(checkpoint, self.ckpt_dir / f'epoch_{epoch+1:03d}.pth')
        
        if is_best:
            torch.save(checkpoint, self.ckpt_dir / 'best.pth')
    
    def train(self):
        print(f"\nStarting enhancement training from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, val_psnr, val_ssim = self.validate()
            
            self.val_losses.append(val_loss)
            self.scheduler.step()
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f} | PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")
            
            if (epoch + 1) % self.args.sample_interval == 0:
                self.save_samples(epoch)
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print(f"\nTraining complete! Best val loss: {self.best_loss:.6f}")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--val-csv', type=str, required=True)
    parser.add_argument('--pretrained-denoiser', type=str, default='',
                       help='Stage-1 denoiser checkpoint (optional)')
    parser.add_argument('--exp-dir', type=str, required=True)
    parser.add_argument('--resume', type=str, default='')
    
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=512)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--l1-weight', type=float, default=0.5)
    parser.add_argument('--ssim-weight', type=float, default=0.2)
    parser.add_argument('--edge-weight', type=float, default=0.2)
    parser.add_argument('--perceptual-weight', type=float, default=0.1)
    
    parser.add_argument('--sample-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = EnhancerTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()