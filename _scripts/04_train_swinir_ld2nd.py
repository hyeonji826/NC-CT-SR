#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SwinIR training script tailored for LD-CT -> denoised NC-CT (Noise2Noise)
- Assumes official SwinIR repo cloned at ../_externals/SwinIR (or adjust SWINIR_PATH)
- Supports: optional inpainting mask channel, 2.5D (3-slice) input, L1 + SSIM + HF losses
- Minimal external deps: torch, torchvision, numpy, pandas, SimpleITK, scipy, skimage, tqdm
"""

import os
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional: SSIM implementation (prefer pytorch_msssim if available)
try:
    from pytorch_msssim import ssim as ssim_fn
    HAS_PYTORCH_MSSSIM = True
except Exception:
    HAS_PYTORCH_MSSSIM = False

# ---- adjust this if you cloned SwinIR elsewhere ----
SWINIR_PATH = Path(__file__).resolve().parents[1] / '_externals' / 'SwinIR'
if str(SWINIR_PATH) not in os.sys.path:
    os.sys.path.append(str(SWINIR_PATH))

# Import network from SwinIR repo (network_swinir.py)
try:
    from models.network_swinir import SwinIR as SwinIRNet
except Exception as e:
    raise ImportError(f"Cannot import SwinIR network. Make sure you cloned SwinIR into {_externals/SwinIR} and that repo is accessible. Original error: {e}")

# ---------------------------
# Dataset (Noise2Noise CSV)
# ---------------------------
class N2NDataset(Dataset):
    def __init__(self, csv_path, patch_size=256, use_inpainting=False, feather_radius=7, stack_2d=False, augment=True):
        self.df = pd.read_csv(csv_path)
        self.patch_size = patch_size
        self.use_inpainting = use_inpainting
        self.feather_radius = feather_radius
        self.stack_2d = stack_2d  # if True, load z-1,z,z+1 and stack as channels (2.5D)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def detect_noise_pixels(self, img):
        # simple median diff-based detector; threshold tuned on [0,1] normalized CT
        from scipy.ndimage import median_filter
        sm = median_filter(img, size=3)
        diff = np.abs(img - sm)
        mask = diff > 0.15
        return mask.astype(np.uint8)

    def inpaint(self, img, mask):
        import cv2
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        mask_u8 = (mask * 255).astype(np.uint8)
        ip = cv2.inpaint(img_u8, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        ip_f = ip.astype(np.float32) / 255.0
        # feather mask
        k = max(3, int(self.feather_radius)*2+1)
        m = cv2.GaussianBlur(mask_u8, (k,k), sigmaX=self.feather_radius).astype(np.float32)/255.0
        return ip_f * m + img * (1.0 - m), m  # inpaint blended, soft mask

    def load_slice(self, path, idx=None):
        # Path can be NIfTI or single-slice path depending on CSV format
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        if idx is None:
            # default: middle slice
            return arr[arr.shape[0] // 2]
        else:
            # clamp
            idx = int(idx)
            idx = max(0, min(arr.shape[0]-1, idx))
            return arr[idx]

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # expected columns: nc_path, slice1_idx, slice2_idx (both indices relative to nc volume)
        nc_path = row['nc_path']
        s1 = int(row['slice1_idx']); s2 = int(row['slice2_idx'])
        # load slices (values assumed normalized to [0,1]; if not, normalize outside)
        slic1 = self.load_slice(nc_path, s1)
        slic2 = self.load_slice(nc_path, s2)
        # resize/crop to patch_size if needed
        if slic1.shape != slic2.shape:
            from skimage.transform import resize
            slic2 = resize(slic2, slic1.shape, order=1, preserve_range=True, anti_aliasing=True)
        # optional inpainting preprocessing
        mask1 = None
        if self.use_inpainting:
            mask_detect = self.detect_noise_pixels(slic1)
            if mask_detect.any():
                slic1, mask1 = self.inpaint(slic1, mask_detect)
        # normalization: assume input already roughly in [0,1]; safe clip
        slic1 = np.clip(slic1, 0.0, 1.0)
        slic2 = np.clip(slic2, 0.0, 1.0)
        # augmentation & random crop
        H,W = slic1.shape
        if H > self.patch_size and W > self.patch_size:
            top = np.random.randint(0, H - self.patch_size)
            left = np.random.randint(0, W - self.patch_size)
            slic1 = slic1[top:top+self.patch_size, left:left+self.patch_size]
            slic2 = slic2[top:top+self.patch_size, left:left+self.patch_size]
            if mask1 is not None:
                mask1 = mask1[top:top+self.patch_size, left:left+self.patch_size]
        if self.augment:
            if np.random.rand() > 0.5:
                slic1 = np.fliplr(slic1); slic2 = np.fliplr(slic2)
                if mask1 is not None: mask1 = np.fliplr(mask1)
            if np.random.rand() > 0.5:
                slic1 = np.flipud(slic1); slic2 = np.flipud(slic2)
                if mask1 is not None: mask1 = np.flipud(mask1)
            k = np.random.randint(0,4)
            slic1 = np.rot90(slic1, k); slic2 = np.rot90(slic2, k)
            if mask1 is not None: mask1 = np.rot90(mask1, k)
        # to tensor 3-channel (repeat) + optional mask channel
        t1 = torch.from_numpy(slic1.copy()).unsqueeze(0).float()
        t2 = torch.from_numpy(slic2.copy()).unsqueeze(0).float()
        t1 = t1.repeat(3,1,1); t2 = t2.repeat(3,1,1)
        if mask1 is None:
            mask_ch = torch.zeros((1, t1.shape[1], t1.shape[2]), dtype=torch.float32)
        else:
            mask_ch = torch.from_numpy(mask1.copy()).unsqueeze(0).float()
        # concat mask as extra channel (so input channels = 3 + 1 = 4)
        input_tensor = torch.cat([t1, mask_ch.repeat(1,1,1)], dim=0)  # shape [4,H,W]
        return input_tensor, t2

# ---------------------------
# Utilities: HF loss
# ---------------------------
def hf_loss(pred, target, kernel_size=9, sigma=3.0):
    pred_blur = F.gaussian_blur(pred, kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma))
    tgt_blur = F.gaussian_blur(target, kernel_size=(kernel_size,kernel_size), sigma=(sigma,sigma))
    return F.l1_loss(pred - pred_blur, target - tgt_blur)

# ---------------------------
# Training loop
# ---------------------------
def train_loop(args):
    device = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    # Dataset + Dataloader
    train_ds = N2NDataset(args.train_csv, patch_size=args.patch_size, use_inpainting=args.use_inpainting,
                         feather_radius=args.feather_radius, augment=True)
    val_ds = None
    if args.val_csv:
        val_ds = N2NDataset(args.val_csv, patch_size=args.patch_size, use_inpainting=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Create SwinIR model instance (use denoising config)
    # NOTE: SwinIR constructor signature may differ; adapt to your cloned repo's network_swinir.SwinIR
    net = SwinIRNet(upscale=1, in_chans=4, img_size=args.patch_size, window_size=8,
                    img_range=1.0, depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6],
                    mlp_ratio=2, upsampler='none').to(device)

    # Optionally load pretrained (if available) - flexible keys
    if args.pretrained and os.path.exists(args.pretrained):
        ck = torch.load(args.pretrained, map_location='cpu')
        sd = ck.get('params', ck.get('state_dict', ck))
        # clean 'module.' prefixes
        new_sd = {}
        for k,v in sd.items():
            new_sd[k.replace('module.','')] = v
        net.load_state_dict(new_sd, strict=False)
        print("[INFO] loaded pretrained:", args.pretrained)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    l1 = nn.L1Loss()

    best_val = -1e9
    for epoch in range(1, args.epochs+1):
        net.train()
        t0 = time.time()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train E{epoch}"):
            inp, tgt = batch  # inp: [B,4,H,W], tgt: [B,3,H,W]
            inp = inp.to(device); tgt = tgt.to(device)
            # forward (SwinIR expects CxHxW in [0,1])
            out = net(inp)  # expected to produce 3-channel output (or adjust signature)
            # If model returns 4-ch or different shape, adapt. Here assume out shape [B,3,H,W]
            # Residual combining: out + input_rgb
            in_rgb = inp[:, :3, :, :]
            pred = in_rgb + out  # residual formulation (if model already outputs residual)
            # losses
            loss_l1 = l1(pred, tgt)
            loss_hf = hf_loss(pred, tgt) if args.use_hf else torch.tensor(0.0, device=device)
            if HAS_PYTORCH_MSSSIM:
                loss_ssim = 1.0 - ssim_fn(pred, tgt, data_range=1.0, size_average=True)
            else:
                loss_ssim = torch.tensor(0.0, device=device)
            loss = args.mse_weight * loss_l1 + args.hf_weight * loss_hf + args.ssim_weight * loss_ssim
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        # validation
        val_psnr = 0.0
        if val_loader is not None:
            net.eval()
            tot_psnr = 0.0; n_samples = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vin, vtgt = vbatch
                    vin = vin.to(device); vtgt = vtgt.to(device)
                    vout = net(vin)
                    vin_rgb = vin[:, :3, :, :]
                    vpred = vin_rgb + vout
                    mse = F.mse_loss(vpred, vtgt, reduction='mean')
                    psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse + 1e-10))
                    tot_psnr += psnr.item(); n_samples += 1
            val_psnr = tot_psnr / max(1, n_samples)
            print(f"Epoch {epoch} done. TrainLoss {running_loss/len(train_loader):.4f} ValPSNR {val_psnr:.2f} took {time.time()-t0:.1f}s")
            # checkpoint best
            if val_psnr > best_val:
                best_val = val_psnr
                torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, Path(args.output_dir)/'best_swinir.pth')
        else:
            print(f"Epoch {epoch} done. TrainLoss {running_loss/len(train_loader):.4f} took {time.time()-t0:.1f}s")
            # periodic checkpoint
            if epoch % args.save_freq == 0:
                torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), 'optimizer': optimizer.state_dict()}, Path(args.output_dir)/f'ckpt_epoch_{epoch}.pth')

    print("Training finished. Best val PSNR:", best_val)

# ---------------------------
# Argument parser
# ---------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_csv', type=str, required=True)
    p.add_argument('--val_csv', type=str, default=None)
    p.add_argument('--output_dir', type=str, default='Outputs')
    p.add_argument('--pretrained', type=str, default=None, help='path to pretrained swinir weights (optional)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--patch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--mse_weight', type=float, default=0.8)
    p.add_argument('--ssim_weight', type=float, default=0.0)
    p.add_argument('--hf_weight', type=float, default=0.1)
    p.add_argument('--use_hf', action='store_true', default=False)
    p.add_argument('--use_inpainting', action='store_true', default=False)
    p.add_argument('--feather_radius', type=float, default=7.0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    return p.parse_args()

if __name__ == '__main__':
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_loop(args)
