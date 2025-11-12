# -*- coding: utf-8 -*-
"""
NAFNet Inference with Alpha Blending
NC → Predicted CE → Enhanced NC (NC tone preserved)
"""

import warnings
warnings.filterwarnings('ignore')

import argparse
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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize as sk_resize
import matplotlib.pyplot as plt


# ============================================================
# NAFNet (copy from training script)
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
# Dataset
# ============================================================
class NCCTDataset(Dataset):
    def __init__(self, root, pairs_csv, split='test', image_size=512):
        self.root = Path(root)
        self.image_size = image_size
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
                self.samples.append({
                    'pid': row['id7'],
                    'nc_path': nc_path,
                    'ce_path': ce_path
                })
        print(f"[{split}] Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        nc_img = sitk.ReadImage(str(sample['nc_path']))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        ce_img = sitk.ReadImage(str(sample['ce_path']))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        D_nc, H, W = nc_arr.shape
        D_ce = ce_arr.shape[0]
        mid_slice_nc = nc_arr[D_nc // 2]
        mid_slice_ce = ce_arr[D_ce // 2]
        nc_tensor = torch.from_numpy(mid_slice_nc).unsqueeze(0).float()
        ce_tensor = torch.from_numpy(mid_slice_ce).unsqueeze(0).float()
        if nc_tensor.shape[1] != self.image_size or nc_tensor.shape[2] != self.image_size:
            nc_tensor = transforms.functional.resize(nc_tensor, (self.image_size, self.image_size), antialias=False)
            ce_tensor = transforms.functional.resize(ce_tensor, (self.image_size, self.image_size), antialias=False)
        return {
            'pid': sample['pid'],
            'nc': nc_tensor,
            'ce_real': ce_tensor
        }


# ============================================================
# Inference
# ============================================================
def inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading NAFNet model...")
    model = NAFNet(
        img_channel=1,
        width=32,
        middle_blk_num=12,
        enc_blk_nums=[2, 2, 4, 8],
        dec_blk_nums=[2, 2, 2, 2]
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"✓ Model loaded from {args.checkpoint}")
    
    # Load dataset
    dataset = NCCTDataset(args.root, args.pairs_csv, split=args.split, image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    
    print(f"\nProcessing with alpha={args.alpha} (0=pure NC, 1=pure CE)...")
    
    for batch in tqdm(dataloader, desc="Inference"):
        pid = batch['pid'][0]
        nc = batch['nc'].to(device)
        ce_real = batch['ce_real'].to(device)
        
        with torch.no_grad():
            # Predict CE
            pred_ce = model(nc)
            
            # Alpha blending: Enhanced NC = alpha * pred_ce + (1-alpha) * nc
            enhanced_nc = args.alpha * pred_ce + (1 - args.alpha) * nc
        
        # To numpy
        nc_np = nc.cpu()[0, 0].numpy()
        pred_ce_np = pred_ce.cpu()[0, 0].numpy()
        enhanced_nc_np = enhanced_nc.cpu()[0, 0].numpy()
        ce_real_np = ce_real.cpu()[0, 0].numpy()
        
        # Metrics
        # 1) Enhanced NC vs NC
        psnr_enh_nc = psnr(nc_np, enhanced_nc_np, data_range=1.0)
        ssim_enh_nc = ssim(nc_np, enhanced_nc_np, data_range=1.0)
        
        # 2) Enhanced NC vs Real CE
        psnr_enh_real = psnr(ce_real_np, enhanced_nc_np, data_range=1.0)
        ssim_enh_real = ssim(ce_real_np, enhanced_nc_np, data_range=1.0)
        
        # 3) Pred CE vs Real CE
        psnr_pred_real = psnr(ce_real_np, pred_ce_np, data_range=1.0)
        ssim_pred_real = ssim(ce_real_np, pred_ce_np, data_range=1.0)
        
        # 4) NC vs Real CE (baseline)
        psnr_nc_real = psnr(nc_np, ce_real_np, data_range=1.0)
        ssim_nc_real = ssim(nc_np, ce_real_np, data_range=1.0)
        
        metrics = {
            'pid': pid,
            'alpha': args.alpha,
            'psnr_enh_nc': psnr_enh_nc,
            'ssim_enh_nc': ssim_enh_nc,
            'psnr_enh_real': psnr_enh_real,
            'ssim_enh_real': ssim_enh_real,
            'psnr_pred_real': psnr_pred_real,
            'ssim_pred_real': ssim_pred_real,
            'psnr_nc_real': psnr_nc_real,
            'ssim_nc_real': ssim_nc_real
        }
        all_metrics.append(metrics)
        
        # Save visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(nc_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original NC', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_nc_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Enhanced NC (α={args.alpha})', fontsize=12, fontweight='bold', color='green')
        axes[1].axis('off')
        
        axes[2].imshow(pred_ce_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Predicted CE', fontsize=12, fontweight='bold', color='blue')
        axes[2].axis('off')
        
        axes[3].imshow(ce_real_np, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('Real CE (GT)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        line1 = f"Enh vs NC: PSNR {psnr_enh_nc:.2f}, SSIM {ssim_enh_nc:.4f}"
        line2 = f"Enh vs Real: PSNR {psnr_enh_real:.2f}, SSIM {ssim_enh_real:.4f}"
        line3 = f"Pred vs Real: PSNR {psnr_pred_real:.2f}, SSIM {ssim_pred_real:.4f}"
        
        fig.text(0.5, 0.08, line1, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        fig.text(0.5, 0.04, line2, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        fig.text(0.5, 0.00, line3, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        save_path = output_dir / f'{pid}_alpha{args.alpha}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save arrays
        np.save(output_dir / f'{pid}_enhanced_nc.npy', enhanced_nc_np)
        np.save(output_dir / f'{pid}_pred_ce.npy', pred_ce_np)
    
    # Save metrics
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(output_dir / 'metrics.csv', index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Inference Results (alpha={args.alpha}):")
    print(f"\n[Enhanced NC vs Original NC]")
    print(f"  PSNR: {df_metrics['psnr_enh_nc'].mean():.2f} ± {df_metrics['psnr_enh_nc'].std():.2f} dB")
    print(f"  SSIM: {df_metrics['ssim_enh_nc'].mean():.4f} ± {df_metrics['ssim_enh_nc'].std():.4f}")
    print(f"\n[Enhanced NC vs Real CE]")
    print(f"  PSNR: {df_metrics['psnr_enh_real'].mean():.2f} ± {df_metrics['psnr_enh_real'].std():.2f} dB")
    print(f"  SSIM: {df_metrics['ssim_enh_real'].mean():.4f} ± {df_metrics['ssim_enh_real'].std():.4f}")
    print(f"\n[Predicted CE vs Real CE]")
    print(f"  PSNR: {df_metrics['psnr_pred_real'].mean():.2f} ± {df_metrics['psnr_pred_real'].std():.2f} dB")
    print(f"  SSIM: {df_metrics['ssim_pred_real'].mean():.4f} ± {df_metrics['ssim_pred_real'].std():.4f}")
    print(f"\n[Baseline: NC vs Real CE]")
    print(f"  PSNR: {df_metrics['psnr_nc_real'].mean():.2f} ± {df_metrics['psnr_nc_real'].std():.2f} dB")
    print(f"  SSIM: {df_metrics['ssim_nc_real'].mean():.4f} ± {df_metrics['ssim_nc_real'].std():.4f}")
    print(f"{'='*80}\n")
    print(f"✓ Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='NAFNet Inference with Alpha Blending')
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--checkpoint', required=True, help='Path to NAFNet checkpoint')
    parser.add_argument('--output-dir', default='Outputs/translations/nafnet_enhanced')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Alpha blending: 0=pure NC (tone preserved), 1=pure CE (max enhancement)')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("NAFNet Inference with Alpha Blending")
    print(f"  Alpha: {args.alpha} (0=NC tone, 1=CE enhancement)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*80}\n")
    
    inference(args)


if __name__ == '__main__':
    main()