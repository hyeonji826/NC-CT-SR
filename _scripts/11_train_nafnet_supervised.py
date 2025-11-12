# -*- coding: utf-8 -*-
"""
Supervised NAFNet Training with Pseudo-pairs
NC → CE 직접 학습 (Pseudo-paired)
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


# NAFNet Architecture (동일)
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
    def __init__(self, img_channel=1, width=64, middle_blk_num=6, enc_blk_nums=[1,1,2,4], dec_blk_nums=[1,1,1,1]):
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


# Pseudo-paired Dataset
class PseudoPairedDataset(Dataset):
    def __init__(self, root, pseudo_pairs_csv, split='train', image_size=512, augment=True):
        self.root = Path(root)
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        df = pd.read_csv(self.root / pseudo_pairs_csv)
        
        # Train/Val/Test split
        n_total = len(df)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        if split == 'train':
            df = df.iloc[:n_train]
        elif split == 'val':
            df = df.iloc[n_train:n_train+n_val]
        else:
            df = df.iloc[n_train+n_val:]
        
        self.pairs = df.reset_index(drop=True)
        print(f"[{split}] Loaded {len(self.pairs)} pseudo-pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        # Load NC slice
        nc_img = sitk.ReadImage(row['nc_path'])
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        nc_slice = nc_arr[int(row['nc_slice_idx'])]
        
        # Load CE slice
        ce_img = sitk.ReadImage(row['ce_path'])
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        ce_slice = ce_arr[int(row['ce_slice_idx'])]
        
        # To tensor
        nc_tensor = torch.from_numpy(nc_slice).unsqueeze(0).float()
        ce_tensor = torch.from_numpy(ce_slice).unsqueeze(0).float()
        
        # Resize
        if nc_tensor.shape[1] != self.image_size or nc_tensor.shape[2] != self.image_size:
            nc_tensor = transforms.functional.resize(nc_tensor, (self.image_size, self.image_size), antialias=False)
            ce_tensor = transforms.functional.resize(ce_tensor, (self.image_size, self.image_size), antialias=False)
        
        # Augmentation
        if self.augment:
            if random.random() > 0.5:
                nc_tensor = transforms.functional.hflip(nc_tensor)
                ce_tensor = transforms.functional.hflip(ce_tensor)
            if random.random() > 0.5:
                nc_tensor = transforms.functional.vflip(nc_tensor)
                ce_tensor = transforms.functional.vflip(ce_tensor)
            if random.random() > 0.5:
                k = random.randint(1, 3)
                nc_tensor = torch.rot90(nc_tensor, k, dims=[1,2])
                ce_tensor = torch.rot90(ce_tensor, k, dims=[1,2])
        
        return {'input': nc_tensor, 'target': ce_tensor}


# Trainer
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
        print(f"Total params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.l1 = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.ssim_w = args.ssim_w
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        
        self.train_dataset = PseudoPairedDataset(args.root, args.pseudo_pairs_csv, 'train', args.image_size, True)
        self.val_dataset = PseudoPairedDataset(args.root, args.pseudo_pairs_csv, 'val', args.image_size, False)
        
        self.train_loader = DataLoader(self.train_dataset, args.batch_size, True, num_workers=args.num_workers, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, args.batch_size, False, num_workers=args.num_workers)
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience = args.patience
        self.patience_counter = 0
    
    def _loss(self, pred, target):
        return self.l1(pred, target) + self.ssim_w * self.ssim_loss(pred, target)
    
    def train_epoch(self, epoch):
        self.model.train()
        total = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch in pbar:
            inp = batch['input'].to(self.device)
            target = batch['target'].to(self.device)
            pred = self.model(inp)
            loss = self._loss(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
            pbar.set_postfix({'loss': loss.item()})
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
        target = sample['target'].unsqueeze(0).to(self.device)
        pred = self.model(inp)
        inp_np = inp.cpu()[0,0].numpy()
        pred_np = pred.cpu()[0,0].numpy()
        target_np = target.cpu()[0,0].numpy()
        psnr_val = psnr(target_np, pred_np, data_range=1.0)
        ssim_val = ssim(target_np, pred_np, data_range=1.0)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(inp_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('NC Input', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Predicted CE', fontsize=14, fontweight='bold', color='green')
        axes[1].axis('off')
        axes[2].imshow(target_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Target CE (Pseudo-paired)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        fig.text(0.5, 0.02, f'PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f} | Epoch: {epoch+1} | Val Loss: {self.best_val_loss:.4f}',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = self.exp_dir / 'samples' / f'best_epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Sample saved: {save_path}")
    
    def save_checkpoint(self, filename):
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(), 'train_losses': self.train_losses,
                   'val_losses': self.val_losses, 'best_val_loss': self.best_val_loss,
                   'patience_counter': self.patience_counter}, self.exp_dir / 'checkpoints' / filename)
    
    def train(self):
        print(f"\n{'='*80}")
        print("Supervised NAFNet Training (NC → CE)")
        print(f"  Method: Pseudo-paired Supervised Learning")
        print(f"  Loss: L1 + SSIM({self.ssim_w})")
        print(f"  Early Stopping: patience={self.patience}")
        print(f"{'='*80}\n")
        for epoch in range(self.args.epochs):
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
                break
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1:03d}.pth')
        print("\n✓ Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pseudo-pairs-csv', default='Data/pseudo_pairs.csv')
    parser.add_argument('--exp-dir', default='Outputs/experiments/nafnet_supervised')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--middle-blk-num', type=int, default=6)
    parser.add_argument('--enc-blk-nums', type=int, nargs='+', default=[1,1,2,4])
    parser.add_argument('--dec-blk-nums', type=int, nargs='+', default=[1,1,1,1])
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ssim-w', type=float, default=0.5)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
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