"""
03_seg_guided_finetune.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
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
import json

sys.path.insert(0, str(Path(__file__).parent))
from models import StructurePreservingStyleTransfer

try:
    from torchvision.models import vgg19, VGG19_Weights
except ImportError:
    vgg19 = None


# ============================================================
# Schedulers
# ============================================================

class StyleWeightScheduler:
    """
    조영 스타일 가중치 스케줄러
    
    Phase 1 (Warmup): 부드럽게 시작 (0 → 50%)
    Phase 2 (Strong): 강한 학습 (50% → 80%)
    Phase 3 (Fine-tune): 세밀하게 조정 (80% → 100%)
    """
    def __init__(self, base_weight, total_epochs, warmup_ratio=0.2, strong_ratio=0.5):
        self.base_weight = base_weight
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
        self.strong_end_epoch = int(total_epochs * strong_ratio)
        
        print(f"\n📊 Style Weight Schedule:")
        print(f"  Warmup (0→50%): epochs 0-{self.warmup_epochs}")
        print(f"  Strong (50%→100%): epochs {self.warmup_epochs}-{self.strong_end_epoch}")
        print(f"  Fine-tune (100%→80%): epochs {self.strong_end_epoch}-{total_epochs}")
    
    def get_weight(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup: 0 → 50%
            ratio = epoch / self.warmup_epochs
            return self.base_weight * (0.5 * ratio)
        
        elif epoch < self.strong_end_epoch:
            # Strong: 50% → 100%
            progress = (epoch - self.warmup_epochs) / (self.strong_end_epoch - self.warmup_epochs)
            return self.base_weight * (0.5 + 0.5 * progress)
        
        else:
            # Fine-tune: 100% → 80% (살짝 낮춰서 과적합 방지)
            progress = (epoch - self.strong_end_epoch) / (self.total_epochs - self.strong_end_epoch)
            return self.base_weight * (1.0 - 0.2 * progress)


class WeightDecayScheduler:
    """
    Weight Decay 스케줄러 (과적합 방지)
    
    초반: 강하게 (0.01)
    중반: 유지 (0.01)
    후반: 약하게 (0.001) - Fine-tuning
    """
    def __init__(self, initial_wd=0.01, final_wd=0.001, total_epochs=200, warmup_ratio=0.3):
        self.initial_wd = initial_wd
        self.final_wd = final_wd
        self.total_epochs = total_epochs
        self.warmup_epochs = int(total_epochs * warmup_ratio)
    
    def get_weight_decay(self, epoch):
        if epoch < self.warmup_epochs:
            return self.initial_wd
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return self.final_wd + (self.initial_wd - self.final_wd) * 0.5 * (1 + np.cos(np.pi * progress))


class WarmupCosineScheduler:
    """
    학습률 Warmup + Cosine Annealing
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class EarlyStopping:
    """
    Early Stopping (과적합 방지)
    """
    def __init__(self, patience=15, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


# ============================================================
# Dataset with Train/Val Split
# ============================================================

class WeightedNCCEDataset(Dataset):
    """NC + CE + Weight Map Dataset"""
    def __init__(self, nc_dir, ce_dir, weight_dir, image_size=256, augment=True, 
                 train_ratio=0.85, is_train=True, seed=42):
        self.nc_dir = Path(nc_dir)
        self.ce_dir = Path(ce_dir)
        self.weight_dir = Path(weight_dir)
        self.image_size = image_size
        self.augment = augment and is_train
        
        # NC 환자 수집
        nc_patients = sorted([p for p in self.nc_dir.iterdir() if p.is_dir()])
        
        # Train/Val split
        random.seed(seed)
        random.shuffle(nc_patients)
        split_idx = int(len(nc_patients) * train_ratio)
        
        if is_train:
            nc_patients = nc_patients[:split_idx]
        else:
            nc_patients = nc_patients[split_idx:]
        
        # NC 슬라이스 수집
        self.nc_slices = []
        for patient_dir in nc_patients:
            patient_id = patient_dir.name
            nc_path = patient_dir / 'NC_norm.nii.gz'
            weight_path = self.weight_dir / patient_id / 'NC_weight_map.nii.gz'
            
            if nc_path.exists() and weight_path.exists():
                img = sitk.ReadImage(str(nc_path))
                num_slices = img.GetSize()[2]
                
                for slice_idx in range(num_slices):
                    self.nc_slices.append({
                        'patient_id': patient_id,
                        'nc_path': nc_path,
                        'weight_path': weight_path,
                        'slice_idx': slice_idx
                    })
        
        # CE 슬라이스 수집 (전체 사용)
        self.ce_slices = []
        ce_patients = sorted([p for p in self.ce_dir.iterdir() if p.is_dir()])
        
        for patient_dir in ce_patients:
            ce_path = patient_dir / 'CE_norm.nii.gz'
            if ce_path.exists():
                img = sitk.ReadImage(str(ce_path))
                num_slices = img.GetSize()[2]
                
                for slice_idx in range(num_slices):
                    self.ce_slices.append({
                        'patient_id': patient_dir.name,
                        'ce_path': ce_path,
                        'slice_idx': slice_idx
                    })
        
        split_name = "Train" if is_train else "Val"
        print(f"{split_name} NC slices: {len(self.nc_slices)}")
        print(f"CE slices (shared): {len(self.ce_slices)}")
    
    def __len__(self):
        return len(self.nc_slices)
    
    def load_slice(self, nii_path, slice_idx):
        img = sitk.ReadImage(str(nii_path))
        arr = sitk.GetArrayFromImage(img)
        return arr[slice_idx]
    
    def augment_slice(self, img, is_weight=False):
        if not self.augment:
            return img
        
        # Flip
        if random.random() > 0.5:
            img = np.fliplr(img)
        
        # Rotate (weight는 nearest)
        if random.random() > 0.5:
            from scipy.ndimage import rotate
            angle = random.uniform(-15, 15)
            order = 0 if is_weight else 1
            img = rotate(img, angle, reshape=False, order=order, mode='nearest')
        
        # Intensity (only for images)
        if not is_weight and random.random() > 0.5:
            factor = random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 1)
        
        # Gaussian noise (only for images)
        if not is_weight and random.random() > 0.3:
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img
    
    def __getitem__(self, idx):
        # NC + Weight
        nc_info = self.nc_slices[idx]
        nc_slice = self.load_slice(nc_info['nc_path'], nc_info['slice_idx'])
        weight_slice = self.load_slice(nc_info['weight_path'], nc_info['slice_idx'])
        
        # CE (unpaired)
        ce_idx = random.randint(0, len(self.ce_slices) - 1)
        ce_info = self.ce_slices[ce_idx]
        ce_slice = self.load_slice(ce_info['ce_path'], ce_info['slice_idx'])
        
        # Same augmentation for NC and weight
        if self.augment:
            seed = random.randint(0, 2**32 - 1)
            
            random.seed(seed)
            nc_slice = self.augment_slice(nc_slice, is_weight=False)
            
            random.seed(seed)
            weight_slice = self.augment_slice(weight_slice, is_weight=True)
            
            # CE 독립적으로
            ce_slice = self.augment_slice(ce_slice, is_weight=False)
        
        # Resize
        from skimage.transform import resize
        nc_slice = resize(nc_slice, (self.image_size, self.image_size),
                         order=1, preserve_range=True, anti_aliasing=True)
        weight_slice = resize(weight_slice, (self.image_size, self.image_size),
                            order=0, preserve_range=True, anti_aliasing=False)
        ce_slice = resize(ce_slice, (self.image_size, self.image_size),
                         order=1, preserve_range=True, anti_aliasing=True)
        
        # To tensor
        nc_tensor = torch.from_numpy(nc_slice).float().unsqueeze(0)
        weight_tensor = torch.from_numpy(weight_slice).float().unsqueeze(0)
        ce_tensor = torch.from_numpy(ce_slice).float().unsqueeze(0)
        
        return {
            'nc': nc_tensor,
            'ce': ce_tensor,
            'weight': weight_tensor,
            'nc_patient': nc_info['patient_id']
        }


# ============================================================
# Improved Loss with Label Smoothing
# ============================================================

class ImprovedWeightedLoss(nn.Module):
    """
    개선된 Loss
    - Content Loss with Label Smoothing
    - Weighted Style Loss
    - Perceptual Loss
    - Total Variation Loss (smoothness)
    """
    def __init__(self, 
                 content_weight=10.0,
                 style_weight=50.0,
                 perceptual_weight=3.0,
                 tv_weight=0.1,
                 smoothing=0.1,
                 device='cuda'):
        super().__init__()
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.perceptual_weight = perceptual_weight
        self.tv_weight = tv_weight
        self.smoothing = smoothing
        self.device = device
        
        # VGG
        if vgg19 is not None:
            vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
            self.vgg = nn.Sequential(*list(vgg.children())[:16])
            for param in self.vgg.parameters():
                param.requires_grad = False
        else:
            self.vgg = None
    
    def content_loss(self, enhanced, original):
        """구조 보존 with Label Smoothing"""
        loss = F.l1_loss(enhanced, original)
        
        # Label smoothing (약간의 허용)
        if self.smoothing > 0:
            smooth_target = original * (1 - self.smoothing) + 0.5 * self.smoothing
            smooth_loss = F.l1_loss(enhanced, smooth_target)
            loss = 0.7 * loss + 0.3 * smooth_loss
        
        return loss
    
    def weighted_style_loss(self, enhanced, ce_ref, weight_map):
        """Weight map 기반 조영 학습"""
        # Intensity difference
        diff = torch.abs(enhanced - ce_ref)
        
        # Weighted
        weighted_diff = diff * weight_map
        
        return weighted_diff.mean()
    
    def perceptual_loss(self, enhanced, original):
        """VGG 기반"""
        if self.vgg is None:
            return torch.tensor(0.0, device=self.device)
        
        enhanced_rgb = enhanced.repeat(1, 3, 1, 1)
        original_rgb = original.repeat(1, 3, 1, 1)
        
        enhanced_feat = self.vgg(enhanced_rgb)
        original_feat = self.vgg(original_rgb)
        
        return F.l1_loss(enhanced_feat, original_feat)
    
    def total_variation_loss(self, img):
        """부드러움 유지"""
        tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
        tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
        return tv_h + tv_w
    
    def forward(self, enhanced, original, ce_ref, weight_map):
        loss_content = self.content_loss(enhanced, original)
        loss_style = self.weighted_style_loss(enhanced, ce_ref, weight_map)
        loss_perceptual = self.perceptual_loss(enhanced, original)
        loss_tv = self.total_variation_loss(enhanced)
        
        total = (
            self.content_weight * loss_content +
            self.style_weight * loss_style +
            self.perceptual_weight * loss_perceptual +
            self.tv_weight * loss_tv
        )
        
        loss_dict = {
            'content': loss_content.item(),
            'style': loss_style.item(),
            'perceptual': loss_perceptual.item(),
            'tv': loss_tv.item(),
            'total': total.item()
        }
        
        return total, loss_dict


# ============================================================
# Advanced Trainer
# ============================================================

class AdvancedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        
        # Directories
        self.exp_dir = Path(args.exp_dir)
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.samples_dir = self.exp_dir / 'samples'
        self.logs_dir = self.exp_dir / 'logs'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Model
        print("\n모델 로딩...")
        self.model = StructurePreservingStyleTransfer(
            base_channels=args.base_channels,
            dropout_rate=args.dropout
        )
        
        # Load Phase 1
        if args.load_checkpoint:
            self.load_phase1_checkpoint(args.load_checkpoint)
        
        self.model = self.model.to(self.device)
        
        # Freeze Content Encoder
        print("\nFreezing Content Encoder...")
        for name, param in self.model.named_parameters():
            if 'content_encoder' in name:
                param.requires_grad = False
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"\n파라미터:")
        print(f"  Total: {total:,}")
        print(f"  Trainable: {trainable:,}")
        print(f"  Frozen: {total - trainable:,}")
        
        # Loss
        self.criterion = ImprovedWeightedLoss(
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            perceptual_weight=args.perceptual_weight,
            tv_weight=args.tv_weight,
            smoothing=args.label_smoothing,
            device=self.device
        )
        
        # Optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Schedulers
        warmup_epochs = int(args.epochs * 0.1)
        self.lr_scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=args.epochs,
            base_lr=args.lr,
            min_lr=args.min_lr
        )
        
        self.style_weight_scheduler = StyleWeightScheduler(
            base_weight=args.style_weight,
            total_epochs=args.epochs,
            warmup_ratio=0.2,
            strong_ratio=0.5
        )
        
        self.wd_scheduler = WeightDecayScheduler(
            initial_wd=args.weight_decay,
            final_wd=args.min_weight_decay,
            total_epochs=args.epochs,
            warmup_ratio=0.3
        )
        
        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=args.patience,
            min_delta=1e-5
        )
        
        # Mixed Precision
        self.scaler = GradScaler() if args.use_amp else None
        
        # Datasets
        print("\n데이터셋 로딩...")
        self.train_dataset = WeightedNCCEDataset(
            nc_dir=args.nc_dir,
            ce_dir=args.ce_dir,
            weight_dir=args.weight_dir,
            image_size=args.image_size,
            augment=True,
            train_ratio=args.train_ratio,
            is_train=True,
            seed=args.seed
        )
        
        self.val_dataset = WeightedNCCEDataset(
            nc_dir=args.nc_dir,
            ce_dir=args.ce_dir,
            weight_dir=args.weight_dir,
            image_size=args.image_size,
            augment=False,
            train_ratio=args.train_ratio,
            is_train=False,
            seed=args.seed
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        self.val_history = []
    
    def load_phase1_checkpoint(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"⚠️ Checkpoint not found: {ckpt_path}")
            return
        
        print(f"Loading Phase 1: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Phase 1 loaded!")
    
    def train_epoch(self, epoch):
        self.model.train()
        
        # Update schedulers
        current_style_weight = self.style_weight_scheduler.get_weight(epoch)
        current_wd = self.wd_scheduler.get_weight_decay(epoch)
        
        # Update weight decay
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = current_wd
        
        # Update style weight in loss
        self.criterion.style_weight = current_style_weight
        
        epoch_losses = {
            'content': [],
            'style': [],
            'perceptual': [],
            'tv': [],
            'total': []
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        accumulation_steps = self.args.grad_accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            nc = batch['nc'].to(self.device)
            ce = batch['ce'].to(self.device)
            weight = batch['weight'].to(self.device)
            
            # Forward
            if self.scaler:
                with autocast():
                    enhanced = self.model(nc, ce, alpha=self.args.style_alpha)
                    loss, loss_dict = self.criterion(enhanced, nc, ce, weight)
                    loss = loss / accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                enhanced = self.model(nc, ce, alpha=self.args.style_alpha)
                loss, loss_dict = self.criterion(enhanced, nc, ce, weight)
                loss = loss / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Record (원래 scale로)
            for key in loss_dict:
                epoch_losses[key].append(loss_dict[key])
            
            pbar.set_postfix({
                'loss': loss_dict['total'],
                'style_w': f'{current_style_weight:.2f}',
                'wd': f'{current_wd:.4f}'
            })
        
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        
        val_losses = {
            'content': [],
            'style': [],
            'perceptual': [],
            'tv': [],
            'total': []
        }
        
        psnr_vals = []
        ssim_vals = []
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            nc = batch['nc'].to(self.device)
            ce = batch['ce'].to(self.device)
            weight = batch['weight'].to(self.device)
            
            enhanced = self.model(nc, ce, alpha=self.args.style_alpha)
            loss, loss_dict = self.criterion(enhanced, nc, ce, weight)
            
            for key in loss_dict:
                val_losses[key].append(loss_dict[key])
            
            # Metrics
            for i in range(nc.shape[0]):
                nc_np = nc[i, 0].cpu().numpy()
                enhanced_np = enhanced[i, 0].cpu().numpy()
                
                psnr_vals.append(psnr_metric(nc_np, enhanced_np, data_range=1.0))
                ssim_vals.append(ssim_metric(nc_np, enhanced_np, data_range=1.0))
        
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        
        return avg_losses, avg_psnr, avg_ssim
    
    @torch.no_grad()
    def save_samples(self, epoch):
        self.model.eval()
        
        batch = next(iter(self.val_loader))
        nc = batch['nc'][:4].to(self.device)
        ce = batch['ce'][:4].to(self.device)
        weight = batch['weight'][:4].to(self.device)
        
        enhanced = self.model(nc, ce, alpha=self.args.style_alpha)
        
        # Metrics
        psnr_vals = []
        ssim_vals = []
        for i in range(nc.shape[0]):
            nc_np = nc[i, 0].cpu().numpy()
            enhanced_np = enhanced[i, 0].cpu().numpy()
            
            psnr_vals.append(psnr_metric(nc_np, enhanced_np, data_range=1.0))
            ssim_vals.append(ssim_metric(nc_np, enhanced_np, data_range=1.0))
        
        # Visualize
        fig, axes = plt.subplots(5, 4, figsize=(16, 20))
        fig.suptitle(
            f'Epoch {epoch+1} - Advanced Seg-Guided\n'
            f'PSNR: {np.mean(psnr_vals):.2f} | SSIM: {np.mean(ssim_vals):.4f}',
            fontsize=14, fontweight='bold'
        )
        
        for i in range(4):
            axes[0, i].imshow(nc[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title('NC', fontweight='bold', color='blue')
            axes[0, i].axis('off')
            
            weight_vis = weight[i, 0].cpu().numpy()
            im = axes[1, i].imshow(weight_vis, cmap='jet', vmin=0.1, vmax=1.0)
            axes[1, i].set_title(f'Weight Map', color='purple')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(ce[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_title('CE (Reference)', color='red')
            axes[2, i].axis('off')
            
            axes[3, i].imshow(enhanced[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[3, i].set_title(f'Enhanced\nPSNR: {psnr_vals[i]:.2f}', color='green')
            axes[3, i].axis('off')
            
            diff = np.abs(enhanced[i, 0].cpu().numpy() - nc[i, 0].cpu().numpy())
            axes[4, i].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
            axes[4, i].set_title(f'SSIM: {ssim_vals[i]:.4f}')
            axes[4, i].axis('off')
        
        plt.colorbar(im, ax=axes[1, :], fraction=0.046, pad=0.04)
        plt.tight_layout()
        save_path = self.samples_dir / f'epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'args': vars(self.args)
        }
        
        ckpt_path = self.ckpt_dir / f'epoch_{epoch+1:03d}.pth'
        torch.save(checkpoint, ckpt_path)
        
        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  ★ Best 모델: {best_path}")
    
    def save_history(self):
        """학습 기록 저장"""
        history = {
            'train': self.train_history,
            'val': self.val_history
        }
        
        with open(self.logs_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history) + 1)
        
        # Total Loss
        axes[0, 0].plot(epochs, [x['total'] for x in self.train_history], label='Train')
        axes[0, 0].plot(epochs, [x['total'] for x in self.val_history], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Content Loss
        axes[0, 1].plot(epochs, [x['content'] for x in self.train_history], label='Train')
        axes[0, 1].plot(epochs, [x['content'] for x in self.val_history], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Content Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Style Loss
        axes[1, 0].plot(epochs, [x['style'] for x in self.train_history], label='Train')
        axes[1, 0].plot(epochs, [x['style'] for x in self.val_history], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Style Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Metrics
        if len(self.val_history) > 0 and 'psnr' in self.val_history[0]:
            axes[1, 1].plot(epochs, [x['psnr'] for x in self.val_history], label='PSNR', color='green')
            ax2 = axes[1, 1].twinx()
            ax2.plot(epochs, [x['ssim'] for x in self.val_history], label='SSIM', color='orange')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('PSNR', color='green')
            ax2.set_ylabel('SSIM', color='orange')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.logs_dir / 'training_curves.png', dpi=150)
        plt.close()
    
    def train(self):
        print("\n" + "="*80)
        print("🚀 Advanced Seg-Guided Fine-Tuning 시작!")
        print("="*80)
        print(f"Total Epochs: {self.args.epochs}")
        print(f"Patience: {self.args.patience}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.args.use_amp}")
        print(f"Gradient Accumulation: {self.args.grad_accumulation}")
        print("="*80)
        
        for epoch in range(self.start_epoch, self.args.epochs):
            # LR update
            current_lr = self.lr_scheduler.step(epoch)
            
            # Train
            train_losses = self.train_epoch(epoch)
            self.train_history.append(train_losses)
            
            # Validate
            val_losses, val_psnr, val_ssim = self.validate(epoch)
            val_losses['psnr'] = val_psnr
            val_losses['ssim'] = val_ssim
            self.val_history.append(val_losses)
            
            # Print
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_losses['total']:.6f}")
            print(f"  Val Loss: {val_losses['total']:.6f}")
            print(f"  Val PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}")
            print(f"  Style Weight: {self.criterion.style_weight:.2f}")
            print(f"  Weight Decay: {self.optimizer.param_groups[0]['weight_decay']:.4f}")
            
            # Save samples
            if (epoch + 1) % self.args.sample_interval == 0:
                self.save_samples(epoch)
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.early_stopping(val_losses['total']):
                print(f"\n⚠️ Early Stopping at epoch {epoch+1}")
                break
            
            # Save history
            self.save_history()
        
        print("\n" + "="*80)
        print("✅ 학습 완료!")
        print(f"Best Val Loss: {self.best_val_loss:.6f}")
        print(f"저장 경로: {self.exp_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Advanced Seg-Guided Fine-Tuning'
    )
    
    # Paths
    parser.add_argument('--nc-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\NC')
    parser.add_argument('--ce-dir', type=str,
                       default=r'E:\LD-CT SR\Data\nii_preproc_norm\CE')
    parser.add_argument('--weight-dir', type=str,
                       default=r'E:\LD-CT SR\Data\weight_maps')
    parser.add_argument('--load-checkpoint', type=str,
                       default=r'E:\LD-CT SR\experiments\nc_to_ce_phase1\checkpoints\best.pth')
    parser.add_argument('--exp-dir', type=str, required=True)
    
    # Model
    parser.add_argument('--base-channels', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                       help='최대 에포크 (Early Stopping 적용)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--min-lr', type=float, default=1e-7)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--min-weight-decay', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--train-ratio', type=float, default=0.85)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--grad-accumulation', type=int, default=2)
    parser.add_argument('--use-amp', action='store_true')
    
    # Loss
    parser.add_argument('--content-weight', type=float, default=10.0)
    parser.add_argument('--style-weight', type=float, default=50.0)
    parser.add_argument('--perceptual-weight', type=float, default=3.0)
    parser.add_argument('--tv-weight', type=float, default=0.1)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--style-alpha', type=float, default=1.0)
    
    # Save
    parser.add_argument('--sample-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    trainer = AdvancedTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()