# -*- coding: utf-8 -*-
"""
NC Enhancement with NAFNet
구조 보존하면서 노이즈 제거 및 화질 개선
(NAFSSR StereoSR 사전학습 가중치 로드 + 파인튜닝)
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
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# NAFNet Architecture
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
    def __init__(self, img_channel=1, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
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
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
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


# ============================================================
# StereoSR pretrained loader (NAFSSR → 1ch NAFNet)
# ============================================================
def _adapt_first_last_conv(src_w, tgt_shape):
    # src_w, tgt: [Cout, Cin, k, k]
    sw = src_w
    # Cin 3->1
    if sw.shape[1] != tgt_shape[1]:
        if sw.shape[1] == 3 and tgt_shape[1] == 1:
            sw = sw.mean(dim=1, keepdim=True)
        else:
            sw = sw[:, :tgt_shape[1]]
    # Cout 3->1 (혹시 모를 경우)
    if sw.shape[0] != tgt_shape[0]:
        if sw.shape[0] == 3 and tgt_shape[0] == 1:
            sw = sw.mean(dim=0, keepdim=True)
        else:
            sw = sw[:tgt_shape[0]]
    return sw

def _maybe_avg_bias(src_b, tgt_shape_out):
    b = src_b
    if b.shape[0] != tgt_shape_out:
        if b.shape[0] == 3 and tgt_shape_out == 1:
            b = b.mean().unsqueeze(0)
        else:
            b = b[:tgt_shape_out]
    return b

def load_stereosr_pretrained(model: torch.nn.Module, ckpt_path: str) -> int:
    """
    StereoSR(NAFSSR) 체크포인트에서 우리 NAFNet(1ch)으로 가능한 파라미터만 로드.
    - 키/shape 일치: 그대로 로드
    - intro/ending conv의 3채널↔1채널은 평균 변환
    - shape 불일치/특수 모듈은 스킵
    return: 로드된 텐서 개수
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for k in ['params_ema', 'params', 'state_dict', 'model', 'net', 'network']:
            if k in ckpt and isinstance(ckpt[k], dict):
                src = ckpt[k]
                break
        else:
            src = ckpt
    else:
        src = ckpt
    src = {k.replace('module.', ''): v for k, v in src.items()}

    dst = model.state_dict()
    loaded = 0
    new_sd = {}

    for k_dst, v_dst in dst.items():
        if k_dst in src:
            v_src = src[k_dst]
            if v_src.shape == v_dst.shape:
                new_sd[k_dst] = v_src
                loaded += 1
            else:
                # intro/ending conv만 채널 보정 시도
                if ('intro.weight' in k_dst or 'ending.weight' in k_dst) and v_src.ndim == 4 and v_dst.ndim == 4:
                    new_sd[k_dst] = _adapt_first_last_conv(v_src, v_dst.shape)
                    loaded += 1
                elif ('intro.bias' in k_dst or 'ending.bias' in k_dst) and v_src.ndim == 1 and v_dst.ndim == 1:
                    new_sd[k_dst] = _maybe_avg_bias(v_src, v_dst.shape[0])
                    loaded += 1
                else:
                    pass
        else:
            pass

    dst.update(new_sd)
    model.load_state_dict(dst, strict=False)
    return loaded


# ============================================================
# Dataset
# ============================================================
class NCEnhancementDataset(Dataset):
    """NC self-supervised dataset"""
    def __init__(self, root, pairs_csv, split='train', image_size=512, num_slices=5):
        self.root = Path(root)
        self.image_size = image_size
        self.num_slices = num_slices
        
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
            ce_path = Path(row['target_ce_norm'])
            if nc_path.exists() and ce_path.exists():
                self.samples.append({'nc_path': nc_path, 'ce_path': ce_path})
        
        print(f"[{split}] Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples) * self.num_slices
    
    def __getitem__(self, idx):
        sample_idx = idx // self.num_slices
        slice_offset = idx % self.num_slices
        
        sample = self.samples[sample_idx]
        
        # NC 로드
        nc_img = sitk.ReadImage(str(sample['nc_path']))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        # CE 로드 (target)
        ce_img = sitk.ReadImage(str(sample['ce_path']))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        
        D_nc, H, W = nc_arr.shape
        D_ce = ce_arr.shape[0]
        
        # 중간 50% 구간에서 균등 샘플
        start_idx = min(D_nc, D_ce) // 4
        end_idx = 3 * min(D_nc, D_ce) // 4
        slice_idx = start_idx + (end_idx - start_idx) * slice_offset // self.num_slices
        slice_idx = np.clip(slice_idx, 0, min(D_nc, D_ce) - 1)
        
        nc_slice = nc_arr[min(slice_idx, D_nc-1)]
        ce_slice = ce_arr[min(slice_idx, D_ce-1)]
        
        # [0,1] 가정
        nc_tensor = torch.from_numpy(nc_slice).unsqueeze(0).float()
        ce_tensor = torch.from_numpy(ce_slice).unsqueeze(0).float()
        
        # Resize
        if nc_tensor.shape[1] != self.image_size or nc_tensor.shape[2] != self.image_size:
            nc_tensor = transforms.functional.resize(nc_tensor, (self.image_size, self.image_size), antialias=True)
            ce_tensor = transforms.functional.resize(ce_tensor, (self.image_size, self.image_size), antialias=True)
        
        return {'nc': nc_tensor, 'ce': ce_tensor}


# ============================================================
# Trainer
# ============================================================
class NAFNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Directories
        self.exp_dir = Path(args.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'samples').mkdir(exist_ok=True)
        
        # Model
        print("Building NAFNet...")
        self.model = NAFNet(
            img_channel=1,
            width=args.width,
            middle_blk_num=args.middle_blk_num,
            enc_blk_nums=args.enc_blk_nums,
            dec_blk_nums=args.dec_blk_nums
        ).to(self.device)
        
        # ===== Load StereoSR (NAFSSR) pretrained if present =====
        if getattr(args, 'pretrained_stereosr', ''):
            ckpt_path = args.pretrained_stereosr
            if not os.path.isfile(ckpt_path):
                print(f"[WARN] Pretrained not found at {ckpt_path}. Skip loading.")
            else:
                n = load_stereosr_pretrained(self.model, ckpt_path)
                print(f"[INFO] Loaded {n} tensors from NAFSSR-{args.pretrained_which} pretrained.")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total params: {total_params:,}")
        
        # Loss
        self.criterion = nn.L1Loss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.001
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )
        
        # Datasets
        self.train_dataset = NCEnhancementDataset(
            args.root, args.pairs_csv, split='train',
            image_size=args.image_size, num_slices=args.num_slices
        )
        self.val_dataset = NCEnhancementDataset(
            args.root, args.pairs_csv, split='val',
            image_size=args.image_size, num_slices=2
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
            num_workers=args.num_workers
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch in pbar:
            nc = batch['nc'].to(self.device)
            ce = batch['ce'].to(self.device)
            
            pred = self.model(nc)
            loss = self.criterion(pred, ce)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            nc = batch['nc'].to(self.device)
            ce = batch['ce'].to(self.device)
            pred = self.model(nc)
            loss = self.criterion(pred, ce)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best.pth')
            self.save_sample_comparison(epoch)
            print(f"✓ Best model saved (val_loss: {avg_loss:.4f})")
        
        return avg_loss
    
    @torch.no_grad()
    def save_sample_comparison(self, epoch):
        """Original NC vs Enhanced NC 비교 이미지 저장"""
        self.model.eval()
        sample = self.val_dataset[0]
        nc = sample['nc'].unsqueeze(0).to(self.device)
        enhanced = self.model(nc)
        
        nc_np = nc.cpu()[0, 0].numpy()
        enhanced_np = enhanced.cpu()[0, 0].numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(nc_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original NC', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Enhanced NC (NAFNet)', fontsize=14, fontweight='bold', color='green')
        axes[1].axis('off')
        
        from skimage.metrics import peak_signal_noise_ratio as psnr
        psnr_val = psnr(nc_np, enhanced_np, data_range=1.0)
        
        fig.text(0.5, 0.02, f'PSNR: {psnr_val:.2f} dB | Epoch: {epoch+1}', 
                 ha='center', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        save_path = self.exp_dir / 'samples' / f'best_epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Sample saved: {save_path}")
    
    def save_checkpoint(self, filename):
        save_path = self.exp_dir / 'checkpoints' / filename
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_path)
    
    def train(self):
        print(f"\n{'='*80}")
        print("Starting NC Enhancement with NAFNet")
        print(f"{'='*80}\n")
        
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1:03d}.pth')
        
        print("\n✓ Training complete!")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='NC Enhancement with NAFNet')
    
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--exp-dir', default='Outputs/experiments/nafnet_enhancement')
    
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--middle-blk-num', type=int, default=12)
    parser.add_argument('--enc-blk-nums', type=int, nargs='+', default=[2, 2, 4, 8])
    parser.add_argument('--dec-blk-nums', type=int, nargs='+', default=[2, 2, 2, 2])
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--num-slices', type=int, default=10)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)

    # ===== 추가: StereoSR(NAFSSR) 프리트레인 경로/라벨 =====
    parser.add_argument('--pretrained-stereosr', default=r'E:\LD-CT SR\Weights\NAFSSR-L_4x.pth',
                        help='NAFSSR ckpt path (.pth). 빈 문자열이면 로드 생략')
    parser.add_argument('--pretrained-which', default='4x', choices=['2x', '4x'],
                        help='로그 표기용')

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
    
    trainer = NAFNetTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
