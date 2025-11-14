#!/usr/bin/env python3
"""
02_train_nc_to_ce.py
Phase 1: NC 구조 보존 + CE 조영 효과 학습

★ 핵심 목표:
1. NC의 해부학적 구조는 절대 보존! (최우선)
2. CE의 조영 패턴만 학습
3. Unpaired 데이터로 학습

Architecture: Structure-Preserving Style Transfer
- Content: NC (구조 절대 보존!)
- Style: CE (조영 패턴만)
- Output: Enhanced NC

Loss 가중치:
- Content Loss: 10.0 (구조 보존 최우선!)
- Style Loss: 1.0 (조영 학습)
- Perceptual Loss: 5.0 (해부학 일관성)
- Anatomy Loss: 3.0 (경계 보존)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import StructurePreservingStyleTransfer

try:
    from pytorch_msssim import ssim as ssim_loss
except ImportError:
    ssim_loss = None

try:
    from torchvision.models import vgg19, VGG19_Weights
except ImportError:
    vgg19 = None


# ============================================================
# Dataset
# ============================================================

class UnpairedNCCEDataset(Dataset):
    """
    Unpaired NC-CE 데이터셋
    
    NC와 CE는 서로 다른 환자일 수 있음 (unpaired)
    각 batch에서 랜덤하게 NC, CE 쌍 생성
    """
    def __init__(self, nc_dir, ce_dir, image_size=256, augment=True):
        self.nc_dir = Path(nc_dir)
        self.ce_dir = Path(ce_dir)
        self.image_size = image_size
        self.augment = augment
        
        # NC 환자 목록
        self.nc_patients = sorted([p for p in self.nc_dir.iterdir() if p.is_dir()])
        self.nc_slices = []
        
        for patient_dir in self.nc_patients:
            nii_path = patient_dir / 'NC_norm.nii.gz'
            if nii_path.exists():
                img = sitk.ReadImage(str(nii_path))
                num_slices = img.GetSize()[2]
                for slice_idx in range(num_slices):
                    self.nc_slices.append({
                        'patient_id': patient_dir.name,
                        'nii_path': nii_path,
                        'slice_idx': slice_idx
                    })
        
        # CE 환자 목록
        self.ce_patients = sorted([p for p in self.ce_dir.iterdir() if p.is_dir()])
        self.ce_slices = []
        
        for patient_dir in self.ce_patients:
            nii_path = patient_dir / 'CE_norm.nii.gz'
            if nii_path.exists():
                img = sitk.ReadImage(str(nii_path))
                num_slices = img.GetSize()[2]
                for slice_idx in range(num_slices):
                    self.ce_slices.append({
                        'patient_id': patient_dir.name,
                        'nii_path': nii_path,
                        'slice_idx': slice_idx
                    })
        
        print(f"NC slices: {len(self.nc_slices)}")
        print(f"CE slices: {len(self.ce_slices)}")
    
    def __len__(self):
        return max(len(self.nc_slices), len(self.ce_slices))
    
    def load_slice(self, nii_path, slice_idx):
        """NIfTI에서 특정 슬라이스 로드"""
        img = sitk.ReadImage(str(nii_path))
        arr = sitk.GetArrayFromImage(img)
        slice_2d = arr[slice_idx]
        return slice_2d
    
    def augment_slice(self, img):
        """Augmentation"""
        if not self.augment:
            return img
        
        # Random flip
        if random.random() > 0.5:
            img = np.fliplr(img)
        
        # Random rotation (small angle)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            from scipy.ndimage import rotate
            img = rotate(img, angle, reshape=False, order=1, mode='nearest')
        
        # Random brightness
        if random.random() > 0.5:
            factor = random.uniform(0.9, 1.1)
            img = np.clip(img * factor, 0, 1)
        
        return img
    
    def __getitem__(self, idx):
        # NC 슬라이스 (구조 기준)
        nc_idx = idx % len(self.nc_slices)
        nc_info = self.nc_slices[nc_idx]
        nc_slice = self.load_slice(nc_info['nii_path'], nc_info['slice_idx'])
        
        # CE 슬라이스 (조영 기준, unpaired - 랜덤)
        ce_idx = random.randint(0, len(self.ce_slices) - 1)
        ce_info = self.ce_slices[ce_idx]
        ce_slice = self.load_slice(ce_info['nii_path'], ce_info['slice_idx'])
        
        # Augmentation
        nc_slice = self.augment_slice(nc_slice)
        ce_slice = self.augment_slice(ce_slice)
        
        # Resize
        from skimage.transform import resize
        nc_slice = resize(nc_slice, (self.image_size, self.image_size), 
                         order=1, preserve_range=True, anti_aliasing=True)
        ce_slice = resize(ce_slice, (self.image_size, self.image_size),
                         order=1, preserve_range=True, anti_aliasing=True)
        
        # To tensor
        nc_tensor = torch.from_numpy(nc_slice).float().unsqueeze(0)
        ce_tensor = torch.from_numpy(ce_slice).float().unsqueeze(0)
        
        return {
            'nc': nc_tensor,              # [1, H, W] - 구조 보존 기준!
            'ce': ce_tensor,              # [1, H, W] - 조영 학습 기준
            'nc_patient': nc_info['patient_id'],
            'ce_patient': ce_info['patient_id']
        }


# ============================================================
# Loss Functions
# ============================================================

class StructurePreservingLoss(nn.Module):
    """
    구조 보존 Loss
    
    Loss 구성:
    1. Content Loss (가장 중요!) - NC 구조 보존
    2. Style Loss - CE 조영 패턴 학습
    3. Perceptual Loss - 해부학적 일관성
    4. Anatomy Loss - 경계 보존
    """
    def __init__(self, 
                 content_weight=10.0,    # ★ 가장 높음! 구조 보존
                 style_weight=1.0,       # 조영 학습
                 perceptual_weight=5.0,  # 해부학 일관성
                 anatomy_weight=3.0,     # 경계 보존
                 device='cuda'):
        super().__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.perceptual_weight = perceptual_weight
        self.anatomy_weight = anatomy_weight
        self.device = device
        
        # VGG for perceptual loss
        if vgg19 is not None:
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
            self.vgg = nn.Sequential(*list(vgg.children())[:16])
            for param in self.vgg.parameters():
                param.requires_grad = False
        else:
            self.vgg = None
        
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def content_loss(self, enhanced_nc, original_nc):
        """
        Content Loss: NC의 구조를 보존했는지 확인
        
        ★ 가장 중요한 Loss! 높은 가중치!
        """
        return F.l1_loss(enhanced_nc, original_nc)
    
    def gram_matrix(self, feat):
        """Gram matrix for style"""
        B, C, H, W = feat.shape
        feat = feat.view(B, C, H * W)
        gram = torch.bmm(feat, feat.transpose(1, 2))
        return gram / (C * H * W)
    
    def style_loss(self, enhanced_nc, ce):
        """
        Style Loss: CE의 조영 패턴 학습
        
        Gram matrix로 조영 통계만 학습 (구조는 X)
        """
        if self.vgg is None:
            # VGG 없으면 간단한 통계 매칭
            enhanced_mean = enhanced_nc.mean(dim=[2, 3], keepdim=True)
            enhanced_std = enhanced_nc.std(dim=[2, 3], keepdim=True)
            ce_mean = ce.mean(dim=[2, 3], keepdim=True)
            ce_std = ce.std(dim=[2, 3], keepdim=True)
            
            loss = F.l1_loss(enhanced_mean, ce_mean) + F.l1_loss(enhanced_std, ce_std)
            return loss
        
        # VGG features
        # Repeat 1-channel to 3-channel for VGG
        enhanced_rgb = enhanced_nc.repeat(1, 3, 1, 1)
        ce_rgb = ce.repeat(1, 3, 1, 1)
        
        enhanced_feat = self.vgg(enhanced_rgb)
        ce_feat = self.vgg(ce_rgb)
        
        # Gram matrix loss
        enhanced_gram = self.gram_matrix(enhanced_feat)
        ce_gram = self.gram_matrix(ce_feat)
        
        loss = F.mse_loss(enhanced_gram, ce_gram)
        return loss
    
    def perceptual_loss(self, enhanced_nc, original_nc):
        """
        Perceptual Loss: 해부학적 일관성 유지
        """
        if self.vgg is None:
            return torch.tensor(0.0, device=self.device)
        
        enhanced_rgb = enhanced_nc.repeat(1, 3, 1, 1)
        original_rgb = original_nc.repeat(1, 3, 1, 1)
        
        enhanced_feat = self.vgg(enhanced_rgb)
        original_feat = self.vgg(original_rgb)
        
        return F.l1_loss(enhanced_feat, original_feat)
    
    def anatomy_loss(self, enhanced_nc, original_nc):
        """
        Anatomy Loss: 장기 경계 보존
        
        Edge map이 유사해야 함 (구조 보존 검증)
        """
        # Enhanced edges
        enhanced_edge_x = F.conv2d(enhanced_nc, self.sobel_x, padding=1)
        enhanced_edge_y = F.conv2d(enhanced_nc, self.sobel_y, padding=1)
        enhanced_edge = torch.sqrt(enhanced_edge_x**2 + enhanced_edge_y**2 + 1e-8)
        
        # Original edges
        original_edge_x = F.conv2d(original_nc, self.sobel_x, padding=1)
        original_edge_y = F.conv2d(original_nc, self.sobel_y, padding=1)
        original_edge = torch.sqrt(original_edge_x**2 + original_edge_y**2 + 1e-8)
        
        return F.l1_loss(enhanced_edge, original_edge)
    
    def forward(self, enhanced_nc, original_nc, ce):
        """
        Total Loss
        
        Args:
            enhanced_nc: 모델 출력 (NC + CE style)
            original_nc: 원본 NC (구조 기준!)
            ce: CE 이미지 (조영 기준)
        """
        loss_content = self.content_loss(enhanced_nc, original_nc)
        loss_style = self.style_loss(enhanced_nc, ce)
        loss_perceptual = self.perceptual_loss(enhanced_nc, original_nc)
        loss_anatomy = self.anatomy_loss(enhanced_nc, original_nc)
        
        total_loss = (
            self.content_weight * loss_content +
            self.style_weight * loss_style +
            self.perceptual_weight * loss_perceptual +
            self.anatomy_weight * loss_anatomy
        )
        
        loss_dict = {
            'content': loss_content.item(),
            'style': loss_style.item(),
            'perceptual': loss_perceptual.item(),
            'anatomy': loss_anatomy.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


# ============================================================
# Trainer
# ============================================================

class NCToCETrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        
        # Directories
        self.exp_dir = Path(args.exp_dir)
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.samples_dir = self.exp_dir / 'samples'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Model
        print("\n모델 초기화...")
        self.model = StructurePreservingStyleTransfer(base_channels=args.base_channels)
        self.model = self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"파라미터 수: {num_params:,}")
        
        # Loss
        self.criterion = StructurePreservingLoss(
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            perceptual_weight=args.perceptual_weight,
            anatomy_weight=args.anatomy_weight,
            device=self.device
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        # Dataset
        print("\n데이터셋 로딩...")
        self.train_dataset = UnpairedNCCEDataset(
            nc_dir=args.nc_dir,
            ce_dir=args.ce_dir,
            image_size=args.image_size,
            augment=True
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        print(f"Batch size: {args.batch_size}")
        print(f"Total batches: {len(self.train_loader)}")
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = {
            'content': [],
            'style': [],
            'perceptual': [],
            'anatomy': [],
            'total': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            nc = batch['nc'].to(self.device)  # 구조 기준
            ce = batch['ce'].to(self.device)  # 조영 기준
            
            # Forward
            enhanced_nc = self.model(nc, ce, alpha=self.args.style_alpha)
            
            # Loss
            loss, loss_dict = self.criterion(enhanced_nc, nc, ce)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Record
            for key in loss_dict:
                epoch_losses[key].append(loss_dict[key])
            
            # Update progress
            pbar.set_postfix({
                'loss': loss_dict['total'],
                'content': loss_dict['content'],
                'style': loss_dict['style']
            })
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        self.train_losses.append(avg_losses)
        
        return avg_losses
    
    @torch.no_grad()
    def save_samples(self, epoch):
        self.model.eval()
        
        # Get one batch
        batch = next(iter(self.train_loader))
        nc = batch['nc'][:4].to(self.device)
        ce = batch['ce'][:4].to(self.device)
        
        # Generate
        enhanced_nc = self.model(nc, ce, alpha=self.args.style_alpha)
        
        # Calculate metrics
        psnr_vals = []
        ssim_vals = []
        for i in range(nc.shape[0]):
            nc_np = nc[i, 0].cpu().numpy()
            enhanced_np = enhanced_nc[i, 0].cpu().numpy()
            
            psnr_vals.append(psnr_metric(nc_np, enhanced_np, data_range=1.0))
            ssim_vals.append(ssim_metric(nc_np, enhanced_np, data_range=1.0))
        
        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        
        # Visualize
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(
            f'Epoch {epoch+1} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}\n'
            f'★ Goal: NC structure preserved + CE contrast added',
            fontsize=14, fontweight='bold'
        )
        
        for i in range(4):
            # Original NC (구조 기준)
            axes[i, 0].imshow(nc[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title('NC (Structure)', fontweight='bold', color='blue')
            axes[i, 0].axis('off')
            
            # CE reference (조영 기준)
            axes[i, 1].imshow(ce[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('CE (Contrast Ref)', color='red')
            axes[i, 1].axis('off')
            
            # Enhanced NC (결과)
            axes[i, 2].imshow(enhanced_nc[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Enhanced NC\nPSNR: {psnr_vals[i]:.2f}', 
                               fontweight='bold', color='green')
            axes[i, 2].axis('off')
            
            # Difference (변화량)
            diff = np.abs(enhanced_nc[i, 0].cpu().numpy() - nc[i, 0].cpu().numpy())
            axes[i, 3].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
            axes[i, 3].set_title(f'Difference\nSSIM: {ssim_vals[i]:.4f}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        save_path = self.samples_dir / f'epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"  샘플 저장: {save_path}")
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
        }
        
        # Regular checkpoint
        ckpt_path = self.ckpt_dir / f'epoch_{epoch+1:03d}.pth'
        torch.save(checkpoint, ckpt_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ★ Best 모델 저장: {best_path}")
    
    def train(self):
        print("\n" + "="*80)
        print("학습 시작!")
        print("="*80)
        print(f"Epochs: {self.args.epochs}")
        print(f"Device: {self.device}")
        print(f"Loss weights:")
        print(f"  - Content (구조 보존): {self.args.content_weight}")
        print(f"  - Style (조영 학습): {self.args.style_weight}")
        print(f"  - Perceptual (해부학): {self.args.perceptual_weight}")
        print(f"  - Anatomy (경계): {self.args.anatomy_weight}")
        print("="*80)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # Train
            losses = self.train_epoch(epoch)
            
            # Step scheduler
            self.scheduler.step()
            
            # Print
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Total Loss: {losses['total']:.6f}")
            print(f"  Content (구조): {losses['content']:.6f}")
            print(f"  Style (조영): {losses['style']:.6f}")
            print(f"  Perceptual: {losses['perceptual']:.6f}")
            print(f"  Anatomy: {losses['anatomy']:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save samples
            if (epoch + 1) % self.args.sample_interval == 0:
                self.save_samples(epoch)
            
            # Save checkpoint
            is_best = losses['total'] < self.best_loss
            if is_best:
                self.best_loss = losses['total']
            
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        print("\n" + "="*80)
        print("학습 완료!")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"모델 저장: {self.ckpt_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: NC Structure Preservation + CE Contrast Learning'
    )
    
    # Paths
    parser.add_argument('--nc-dir', type=str, 
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC')
    parser.add_argument('--ce-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\CE')
    parser.add_argument('--exp-dir', type=str, required=True,
                       help='실험 디렉토리')
    
    # Model
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=256)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--content-weight', type=float, default=10.0,
                       help='구조 보존 (최우선!)')
    parser.add_argument('--style-weight', type=float, default=1.0,
                       help='조영 학습')
    parser.add_argument('--perceptual-weight', type=float, default=5.0,
                       help='해부학 일관성')
    parser.add_argument('--anatomy-weight', type=float, default=3.0,
                       help='경계 보존')
    parser.add_argument('--style-alpha', type=float, default=1.0,
                       help='Style strength (0-1)')
    
    # Save
    parser.add_argument('--sample-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train
    trainer = NCToCETrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()