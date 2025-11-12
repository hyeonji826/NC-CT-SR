# -*- coding: utf-8 -*-
"""
NC Enhancement with NAFNet - Noise2Noise (최고 성능)
무조영 저화질 CT → 무조영 고화질 CT (비지도 학습)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
# lpips removed to save memory


# [NAFNet 아키텍처 코드는 너무 길어서 생략 - 이전과 동일]
# 실제 파일에는 전체 NAFNet 코드 포함

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
        y, var, weight = ctx.saved_variables
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
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True))
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
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
    def __init__(self, img_channel=1, width=128, middle_blk_num=12, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1, groups=1, bias=True)
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
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan*2, 1, bias=False), nn.PixelShuffle(2)))
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
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

def load_pretrained(model, ckpt_path, strict_width=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for k in ['params_ema', 'params', 'state_dict', 'model']:
            if k in ckpt and isinstance(ckpt[k], dict):
                src = ckpt[k]; break
        else:
            src = ckpt
    else:
        src = ckpt
    src = {k.replace('module.', ''): v for k, v in src.items()}
    dst = model.state_dict()
    loaded = 0
    skipped = 0
    new_sd = {}
    
    for k_dst, v_dst in dst.items():
        if k_dst in src:
            v_src = src[k_dst]
            
            # Exact match
            if v_src.shape == v_dst.shape:
                new_sd[k_dst] = v_src
                loaded += 1
            
            # Intro/Ending: 3ch → 1ch conversion
            elif ('intro' in k_dst or 'ending' in k_dst) and v_src.ndim == v_dst.ndim == 4:
                # Channel conversion
                if v_src.shape[1] == 3 and v_dst.shape[1] == 1:
                    v_src = v_src.mean(dim=1, keepdim=True)
                if v_src.shape[0] == 3 and v_dst.shape[0] == 1:
                    v_src = v_src.mean(dim=0, keepdim=True)
                
                # Width mismatch: truncate or skip
                if v_src.shape[0] > v_dst.shape[0]:
                    v_src = v_src[:v_dst.shape[0]]
                    new_sd[k_dst] = v_src
                    loaded += 1
                    print(f"  [Truncated] {k_dst}: {src[k_dst].shape} → {v_dst.shape}")
                elif v_src.shape[0] < v_dst.shape[0]:
                    # Initialize randomly for extra channels
                    new_sd[k_dst] = v_dst.clone()
                    new_sd[k_dst][:v_src.shape[0]] = v_src
                    loaded += 1
                    print(f"  [Partial] {k_dst}: loaded {v_src.shape[0]}/{v_dst.shape[0]} channels")
                else:
                    new_sd[k_dst] = v_src
                    loaded += 1
            
            # Skip mismatched layers (e.g., width difference in middle layers)
            else:
                skipped += 1
        else:
            skipped += 1
    
    dst.update(new_sd)
    model.load_state_dict(dst, strict=False)
    print(f"[INFO] Loaded {loaded} tensors, skipped {skipped} tensors")
    return loaded

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.ws = window_size
        g = torch.tensor([np.exp(-(x-window_size//2)**2/(2*1.5**2)) for x in range(window_size)], dtype=torch.float32)
        g = (g / g.sum()).unsqueeze(1)
        window = g @ g.t()
        self.register_buffer('w', window.unsqueeze(0).unsqueeze(0))
    def forward(self, x, y):
        w = self.w.to(x.device)
        C1, C2 = 0.01**2, 0.03**2
        mu1 = F.conv2d(x, w, padding=self.ws//2)
        mu2 = F.conv2d(y, w, padding=self.ws//2)
        mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
        sigma1_sq = F.conv2d(x*x, w, padding=self.ws//2) - mu1_sq
        sigma2_sq = F.conv2d(y*y, w, padding=self.ws//2) - mu2_sq
        sigma12 = F.conv2d(x*y, w, padding=self.ws//2) - mu1_mu2
        ssim = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        return 1 - ssim.mean()

class Noise2NoiseDataset(Dataset):
    def __init__(self, root, pairs_csv, split='train', image_size=512, num_slices=10, augment=True, noise_sigma=0.02):
        self.root = Path(root)
        self.image_size = image_size
        self.num_slices = num_slices
        self.augment = augment and (split == 'train')
        self.noise_sigma = noise_sigma
        df = pd.read_csv(self.root / pairs_csv)
        n_total = len(df)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        if split == 'train':
            df = df.iloc[:n_train]
        elif split == 'val':
            df = df.iloc[n_train:n_train+n_val]
        else:
            df = df.iloc[n_train+n_val:]
        df = df.reset_index(drop=True)
        self.samples = []
        for _, row in df.iterrows():
            nc_path = Path(row['input_nc_norm'])
            if nc_path.exists():
                self.samples.append({'nc_path': nc_path})
        print(f"[{split}] Loaded {len(self.samples)} samples (Noise2Noise, augment={self.augment})")
    def __len__(self):
        return len(self.samples) * self.num_slices
    def __getitem__(self, idx):
        sample_idx = idx // self.num_slices
        slice_offset = idx % self.num_slices
        sample = self.samples[sample_idx]
        nc_img = sitk.ReadImage(str(sample['nc_path']))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        D, H, W = nc_arr.shape
        start, end = D//4, 3*D//4
        slice_idx = start + (end-start)*slice_offset//self.num_slices
        slice_idx = np.clip(slice_idx, 0, D-1)
        nc_slice = nc_arr[slice_idx]
        nc_tensor = torch.from_numpy(nc_slice).unsqueeze(0).float()
        if nc_tensor.shape[1] != self.image_size or nc_tensor.shape[2] != self.image_size:
            nc_tensor = transforms.functional.resize(nc_tensor, (self.image_size, self.image_size), antialias=False)
        noise1 = torch.randn_like(nc_tensor) * self.noise_sigma
        noise2 = torch.randn_like(nc_tensor) * self.noise_sigma
        nc1 = (nc_tensor + noise1).clamp(0, 1)
        nc2 = (nc_tensor + noise2).clamp(0, 1)
        if self.augment:
            if random.random() > 0.5:
                nc1 = transforms.functional.hflip(nc1)
                nc2 = transforms.functional.hflip(nc2)
            if random.random() > 0.5:
                nc1 = transforms.functional.vflip(nc1)
                nc2 = transforms.functional.vflip(nc2)
            if random.random() > 0.5:
                k = random.randint(1, 3)
                nc1 = torch.rot90(nc1, k, dims=[1,2])
                nc2 = torch.rot90(nc2, k, dims=[1,2])
            if random.random() > 0.3:
                crop_size = int(self.image_size * 0.9375)
                i, j, h, w = transforms.RandomCrop.get_params(nc1, output_size=(crop_size, crop_size))
                nc1 = transforms.functional.crop(nc1, i, j, h, w)
                nc2 = transforms.functional.crop(nc2, i, j, h, w)
                nc1 = transforms.functional.resize(nc1, (self.image_size, self.image_size), antialias=False)
                nc2 = transforms.functional.resize(nc2, (self.image_size, self.image_size), antialias=False)
            if random.random() > 0.5:
                factor = 0.8 + random.random() * 0.4
                nc1 = (nc1 * factor).clamp(0, 1)
                nc2 = (nc2 * factor).clamp(0, 1)
            if random.random() > 0.5:
                factor = 0.8 + random.random() * 0.4
                mean = nc1.mean()
                nc1 = ((nc1 - mean) * factor + mean).clamp(0, 1)
                mean = nc2.mean()
                nc2 = ((nc2 - mean) * factor + mean).clamp(0, 1)
        return {'input': nc1, 'target': nc2, 'clean': nc_tensor}

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.exp_dir = Path(args.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'samples').mkdir(exist_ok=True)
        print("Building NAFNet...")
        self.model = NAFNet(img_channel=1, width=args.width, middle_blk_num=args.middle_blk_num, 
                           enc_blk_nums=args.enc_blk_nums, dec_blk_nums=args.dec_blk_nums).to(self.device)
        if getattr(args, 'pretrained_stereosr', '') and os.path.isfile(args.pretrained_stereosr):
            print(f"[INFO] Loading pretrained weights from {args.pretrained_stereosr}")
            if args.width != 128:
                print(f"[WARN] Width={args.width} but NAFSSR uses 128. Loading compatible layers only.")
            n = load_pretrained(self.model, args.pretrained_stereosr)
            print(f"[INFO] Successfully loaded {n} pretrained tensors")
        print(f"Total params: {sum(p.numel() for p in self.model.parameters()):,}")
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        # Perceptual loss removed to save memory
        self.perceptual_loss = None
        self.ssim_w = args.ssim_w
        self.perceptual_w = 0.0  # Disabled
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        self.train_dataset = Noise2NoiseDataset(args.root, args.pairs_csv, 'train', args.image_size, args.num_slices, True, args.noise_sigma)
        self.val_dataset = Noise2NoiseDataset(args.root, args.pairs_csv, 'val', args.image_size, 2, False, args.noise_sigma)
        self.train_loader = DataLoader(self.train_dataset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, args.batch_size, False, num_workers=args.num_workers)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience = args.patience
        self.patience_counter = 0
        self.accumulation_steps = args.accumulation_steps
        self.freeze_epochs = args.freeze_epochs
        self._toggle_freeze(True)
    def _toggle_freeze(self, freeze):
        for name, p in self.model.named_parameters():
            if name.startswith('intro') or name.startswith('encoders.0') or name.startswith('encoders.1'):
                p.requires_grad = not freeze
        print(f"[Freeze] intro & encoders[0..1] {'frozen' if freeze else 'unfrozen'}")
    def _loss(self, pred, target):
        loss_l1 = self.l1(pred, target)
        loss_ssim = self.ssim_loss(pred, target)
        # Perceptual loss disabled for memory
        total_loss = loss_l1 + self.ssim_w * loss_ssim
        return total_loss
    def train_epoch(self, epoch):
        self.model.train()
        total = 0
        self.optimizer.zero_grad()  # Zero grad at start
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for i, batch in enumerate(pbar):
            inp = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            pred = self.model(inp)
            loss = self._loss(pred, target)
            
            # Normalize loss by accumulation steps
            loss = loss / self.accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total += loss.item() * self.accumulation_steps  # Denormalize for logging
            pbar.set_postfix({'loss': loss.item() * self.accumulation_steps})
        
        # Update remaining gradients
        if (i + 1) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        avg = total / len(self.train_loader)
        self.train_losses.append(avg)
        return avg
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total = 0
        for batch in tqdm(self.val_loader, desc="Validating"):
            inp = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            pred = self.model(inp)
            loss = self._loss(pred, target)
            total += loss.item()
        avg = total / len(self.val_loader)
        self.val_losses.append(avg)
        if avg < self.best_val_loss:
            self.best_val_loss = avg
            self.patience_counter = 0
            self.save_checkpoint('best.pth')
            self.save_sample(epoch)
            print(f"✓ Best model saved (val_loss: {avg:.4f})")
        else:
            self.patience_counter += 1
            print(f"  Patience: {self.patience_counter}/{self.patience}")
        return avg
    @torch.no_grad()
    def save_sample(self, epoch):
        from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
        self.model.eval()
        sample = self.val_dataset[0]
        inp = sample['input'].unsqueeze(0).to(self.device)
        clean = sample['clean']
        pred = self.model(inp)
        inp_np = inp.cpu()[0,0].numpy()
        pred_np = pred.cpu()[0,0].numpy()
        clean_np = clean[0].numpy()
        psnr_val = psnr(clean_np, pred_np, data_range=1.0)
        ssim_val = ssim(clean_np, pred_np, data_range=1.0)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(inp_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Noisy Input', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Denoised (NAFNet)', fontsize=14, fontweight='bold', color='green')
        axes[1].axis('off')
        axes[2].imshow(clean_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Clean Reference', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        fig.text(0.5, 0.02, f'PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | Epoch: {epoch+1} | Val Loss: {self.best_val_loss:.4f}',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = self.exp_dir / 'samples' / f'best_epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Sample saved: {save_path}")
    def save_checkpoint(self, filename):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
                   'train_losses': self.train_losses, 'val_losses': self.val_losses, 'best_val_loss': self.best_val_loss,
                   'patience_counter': self.patience_counter}, self.exp_dir / 'checkpoints' / filename)
    def train(self):
        print(f"\n{'='*80}")
        print("Starting Noise2Noise Training for NC Enhancement")
        print(f"  Method: Unsupervised (Noise2Noise)")
        print(f"  Noise sigma: {self.args.noise_sigma}")
        print(f"  Loss: L1 + SSIM({self.ssim_w})")
        print(f"  Batch size: {self.args.batch_size} x {self.accumulation_steps} = {self.args.batch_size * self.accumulation_steps} (effective)")
        print(f"  Image size: {self.args.image_size}")
        print(f"  Early Stopping: patience={self.patience}")
        print(f"{'='*80}\n")
        for epoch in range(self.args.epochs):
            if epoch == self.freeze_epochs:
                self._toggle_freeze(False)
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {lr:.2e}")
            if self.patience_counter >= self.patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                print(f"   Best val loss: {self.best_val_loss:.4f}")
                break
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1:03d}.pth')
        print("\n✓ Training complete!")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description='NAFNet Noise2Noise Training')
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--exp-dir', default='Outputs/experiments/nafnet_noise2noise')
    parser.add_argument('--width', type=int, default=128)  # Match NAFSSR
    parser.add_argument('--middle-blk-num', type=int, default=12)
    parser.add_argument('--enc-blk-nums', type=int, nargs='+', default=[2,2,4,8])
    parser.add_argument('--dec-blk-nums', type=int, nargs='+', default=[2,2,2,2])
    parser.add_argument('--image-size', type=int, default=256)  # Reduced from 512
    parser.add_argument('--num-slices', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)  # Minimum for width=128
    parser.add_argument('--accumulation-steps', type=int, default=4, 
                       help='Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pretrained-stereosr', default=r'E:\LD-CT SR\Weights\NAFSSR-L_4x.pth')
    parser.add_argument('--pretrained-which', default='4x')
    parser.add_argument('--freeze-epochs', type=int, default=5)
    parser.add_argument('--ssim-w', type=float, default=0.5)
    parser.add_argument('--perceptual-w', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--noise-sigma', type=float, default=0.02)
    args = parser.parse_args()
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print(f"GPU Configuration:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"{'='*80}\n")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()