#!/usr/bin/env python3
"""
Improved NAFNet Inpainting-based Training Script
- ICNR init for sub-pixel convs + PixelShuffle -> conv smoothing
- Option to use Upsample+Conv alternative
- VGG perceptual with ImageNet normalization
- Residual prediction preserved
- High-frequency (HF) loss and SSIM toggle
- Inpainting: optional feather blending for mask boundaries
- Safer defaults for loss weights (L1-centric)
"""
import os
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
from piq import ssim as piq_ssim
# Optional: try import for SSIM (if available). If not, fallback to simple SSIM implementation.
try:
    from piq import ssim as piq_ssim  # piq is optional; prefer if installed
    HAS_PIQ = True
except Exception:
    HAS_PIQ = False

# --------------------------- Utilities ---------------------------

def icnr_init(tensor, upscale_factor=2, init=nn.init.kaiming_normal_):
    """ICNR initialization for sub-pixel conv weights.
    tensor: weight tensor of shape [out_ch, in_ch, k, k]
    upscale_factor: pixelshuffle scale
    """
    with torch.no_grad():
        out_ch, in_ch, k1, k2 = tensor.shape
        new_out = out_ch // (upscale_factor ** 2)
        if new_out <= 0:
            init(tensor)
            return
        subkernel = torch.zeros((new_out, in_ch, k1, k2), dtype=tensor.dtype, device=tensor.device)
        init(subkernel)
        subkernel = subkernel.repeat(upscale_factor ** 2, 1, 1, 1)
        tensor.copy_(subkernel)


def gaussian_feather_mask(mask, radius=7):
    """Feather mask edges with a Gaussian kernel (mask in [0,255] uint8).
    Returns float mask in [0,1]."""
    import cv2
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
    ksize = max(3, int(radius) * 2 + 1)
    blurred = cv2.GaussianBlur(mask_uint8, (ksize, ksize), sigmaX=radius)
    return (blurred.astype(np.float32) / 255.0)

# --------------------------- Losses ---------------------------

class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss with ImageNet normalization for consistency.
    For grayscale CT we repeat channels to 3 and normalize using ImageNet mean/std.
    """
    def __init__(self, device, use_gpu_norm=True):
        super().__init__()
        self.device = device
        # Load pretrained VGG features (eval)
        vgg = models.vgg16(pretrained=True).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.layers = nn.ModuleList([
            vgg[:4].eval(),
            vgg[:9].eval(),
            vgg[:16].eval(),
        ])
        self.weights = [1.0, 0.75, 0.5]
        # ImageNet stats
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def forward(self, pred, target):
        # pred/target: [B,3,H,W] in [0,1]
        # Normalize using ImageNet mean/std
        pred_n = (pred - self.im_mean) / self.im_std
        target_n = (target - self.im_mean) / self.im_std
        loss = 0.0
        for layer, w in zip(self.layers, self.weights):
            pred_feat = layer(pred_n)
            target_feat = layer(target_n)
            loss = loss + w * F.mse_loss(pred_feat, target_feat)
        return loss


class EdgeLoss(nn.Module):
    """Sobel + Laplacian edge loss. Works on single-channel.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[-1, -2, -1], [0,0,0], [1,2,1]], dtype=torch.float32).view(1,1,3,3)
        lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('lap', lap)

    def forward(self, pred, target):
        # pred/target: [B,3,H,W] -> take first channel
        pred_g = pred[:, :1, :, :]
        target_g = target[:, :1, :, :]
        sx = F.conv2d(pred_g, self.sobel_x, padding=1)
        sy = F.conv2d(pred_g, self.sobel_y, padding=1)
        pred_s = torch.sqrt(sx**2 + sy**2 + 1e-6)
        tx = F.conv2d(target_g, self.sobel_x, padding=1)
        ty = F.conv2d(target_g, self.sobel_y, padding=1)
        tgt_s = torch.sqrt(tx**2 + ty**2 + 1e-6)
        pred_l = torch.abs(F.conv2d(pred_g, self.lap, padding=1))
        tgt_l = torch.abs(F.conv2d(target_g, self.lap, padding=1))
        loss_s = F.l1_loss(pred_s, tgt_s)
        loss_l = F.l1_loss(pred_l, tgt_l)
        return 0.7 * loss_s + 0.3 * loss_l


def hf_loss(pred, target, kernel_size=9, sigma=3.0):
    """High-frequency loss computed as L1 between (img - blurred(img))."""
    pred_blur = F.gaussian_blur(pred, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
    tgt_blur = F.gaussian_blur(target, kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
    pred_h = pred - pred_blur
    tgt_h = target - tgt_blur
    return F.l1_loss(pred_h, tgt_h)

# --------------------------- NAFNet blocks (unchanged but minor fixes) ---------------------------

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channel // 2, dw_channel // 2, 1,1,0,bias=True))
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1,1,0,bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c,1,1,0,bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate>0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate>0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1,c,1,1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1,c,1,1)), requires_grad=True)

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


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=12,
                 enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2],
                 use_pixelshuffle=True, upsample_mode='pixelshuffle'):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3,1,1,bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3,1,1,bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2,2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        # Upsampling: either PixelShuffle + conv smoothing or Upsample+Conv
        self.use_pixelshuffle = use_pixelshuffle
        for num in dec_blk_nums:
            if self.use_pixelshuffle and upsample_mode == 'pixelshuffle':
                conv = nn.Conv2d(chan, chan * 4, 3,1,1, bias=True)
                icnr_init(conv.weight, upscale_factor=2)
                up = nn.Sequential(conv, nn.PixelShuffle(2), nn.Conv2d(chan, chan//2, 3,1,1), nn.ReLU(inplace=True))
                chan = chan // 2
                self.ups.append(up)
                self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            else:
                up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                   nn.Conv2d(chan, chan//2, 3,1,1), nn.ReLU(inplace=True))
                chan = chan // 2
                self.ups.append(up)
                self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B,C,H,W = inp.shape
        x = self.check_image_size(inp)
        x = self.intro(x)
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
        x = x + inp  # residual
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _,_,h,w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

# --------------------------- Dataset ---------------------------

class Noise2NoiseDataset(Dataset):
    def __init__(self, csv_path, patch_size=256, augment=True, use_inpainting=True, feather_radius=7):
        self.csv_path = Path(csv_path)
        self.pairs_df = pd.read_csv(csv_path)
        self.patch_size = patch_size
        self.augment = augment
        self.use_inpainting = use_inpainting
        self.feather_radius = feather_radius
        print(f"Loaded {len(self.pairs_df)} Noise2Noise pairs from {csv_path}")

    def detect_noise_pixels(self, slice_2d, threshold=0.15):
        from scipy.ndimage import median_filter
        smoothed = median_filter(slice_2d, size=3)
        diff = np.abs(slice_2d - smoothed)
        noise_mask = diff > threshold
        return noise_mask

    def inpaint_noise(self, slice_2d, noise_mask):
        import cv2
        img_uint8 = np.clip(slice_2d * 255.0, 0, 255).astype(np.uint8)
        mask_uint8 = (noise_mask * 255).astype(np.uint8)
        inpainted = cv2.inpaint(img_uint8, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted.astype(np.float32) / 255.0, mask_uint8

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        import SimpleITK as sitk
        nc_img = sitk.ReadImage(row['nc_path'])
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        slice1 = nc_arr[int(row['slice1_idx'])]
        slice2 = nc_arr[int(row['slice2_idx'])]
        if slice2.shape != slice1.shape:
            from skimage.transform import resize
            slice2 = resize(slice2, slice1.shape, order=1, preserve_range=True, anti_aliasing=True)
        if self.use_inpainting:
            noise_mask1 = self.detect_noise_pixels(slice1)
            if noise_mask1.any():
                slice1_ip, mask1 = self.inpaint_noise(slice1, noise_mask1)
                m1 = gaussian_feather_mask(mask1, radius=self.feather_radius)
                slice1 = slice1_ip * m1 + slice1 * (1 - m1)
            noise_mask2 = self.detect_noise_pixels(slice2)
            if noise_mask2.any():
                slice2_ip, mask2 = self.inpaint_noise(slice2, noise_mask2)
                m2 = gaussian_feather_mask(mask2, radius=self.feather_radius)
                slice2 = slice2_ip * m2 + slice2 * (1 - m2)
        H,W = slice1.shape
        if H > self.patch_size and W > self.patch_size:
            top = np.random.randint(0, H - self.patch_size)
            left = np.random.randint(0, W - self.patch_size)
            slice1 = slice1[top:top+self.patch_size, left:left+self.patch_size]
            slice2 = slice2[top:top+self.patch_size, left:left+self.patch_size]
        if self.augment:
            if np.random.rand() > 0.5:
                slice1 = np.fliplr(slice1); slice2 = np.fliplr(slice2)
            if np.random.rand() > 0.5:
                slice1 = np.flipud(slice1); slice2 = np.flipud(slice2)
            k = np.random.randint(0,4)
            slice1 = np.rot90(slice1, k); slice2 = np.rot90(slice2, k)
            # small subpixel jitter
            if np.random.rand() < 0.5:
                from scipy.ndimage import shift
                sx, sy = (np.random.uniform(-1,1), np.random.uniform(-1,1))
                slice1 = shift(slice1, shift=(sx, sy), mode='reflect')
                slice2 = shift(slice2, shift=(sx, sy), mode='reflect')
        slice1 = np.clip(slice1, 0, 1)
        slice2 = np.clip(slice2, 0, 1)
        t1 = torch.from_numpy(slice1.copy()).unsqueeze(0).float()
        t2 = torch.from_numpy(slice2.copy()).unsqueeze(0).float()
        t1 = t1.repeat(3,1,1); t2 = t2.repeat(3,1,1)
        return t1, t2

# --------------------------- Metrics & utils ---------------------------

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# --------------------------- Checkpoint helper (non-interactive) ---------------------------

def load_checkpoint_if_exists(checkpoint_dir, model, optimizer=None, scheduler=None, resume=False):
    best_checkpoint = checkpoint_dir / 'best_model.pth'
    if best_checkpoint.exists() and resume:
        checkpoint = torch.load(best_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resumed from {best_checkpoint} epoch {start_epoch-1}")
        return start_epoch, best_psnr
    return 1, 0

# --------------------------- Training / Validation ---------------------------

def train_epoch(model, dataloader, criterion_dict, optimizer, device, epoch, args):
    model.train()
    totals = {k:0.0 for k in ['loss','mse','edge','hf','percept','psnr']}
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for noisy1, noisy2 in pbar:
        noisy1 = noisy1.to(device); noisy2 = noisy2.to(device)
        denoised = model(noisy1)
        loss_mse = criterion_dict['mse'](denoised, noisy2)
        loss_edge = criterion_dict['edge'](denoised, noisy2) if 'edge' in criterion_dict else torch.tensor(0.0, device=device)
        loss_hf = hf_loss(denoised, noisy2) if args.use_hf else torch.tensor(0.0, device=device)
        loss_perc = criterion_dict['perc'](denoised, noisy2) if 'perc' in criterion_dict else torch.tensor(0.0, device=device)
        loss = args.mse_weight * loss_mse + args.edge_weight * loss_edge + args.hf_weight * loss_hf + args.perc_weight * loss_perc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        psnr = calculate_psnr(denoised, noisy2)
        totals['loss'] += loss.item(); totals['mse'] += loss_mse.item(); totals['edge'] += (loss_edge.item() if isinstance(loss_edge, torch.Tensor) else 0)
        totals['hf'] += (loss_hf.item() if isinstance(loss_hf, torch.Tensor) else 0); totals['percept'] += (loss_perc.item() if isinstance(loss_perc, torch.Tensor) else 0)
        totals['psnr'] += psnr.item()
        pbar.set_postfix({'loss':f"{loss.item():.4f}", 'psnr':f"{psnr.item():.2f}"})
    n = len(dataloader)
    return totals['loss']/n, totals['mse']/n, totals['edge']/n, totals['psnr']/n

def validate(model, dataloader, criterion_dict, device, args, epoch=None, sample_dir=None):
    model.eval()
    total_loss = 0.0; total_psnr = 0.0
    save_samples = (epoch is not None and sample_dir is not None and epoch % 5 == 0)
    saved_count = 0; max_samples = 3
    with torch.no_grad():
        for batch_idx, (noisy1, noisy2) in enumerate(tqdm(dataloader, desc='Validating')):
            noisy1 = noisy1.to(device); noisy2 = noisy2.to(device)
            denoised = model(noisy1)
            loss = criterion_dict['mse'](denoised, noisy2)
            if 'edge' in criterion_dict:
                loss = loss + args.edge_weight * criterion_dict['edge'](denoised, noisy2)
            if 'perc' in criterion_dict:
                loss = loss + args.perc_weight * criterion_dict['perc'](denoised, noisy2)
            psnr = calculate_psnr(denoised, noisy2)
            total_loss += loss.item(); total_psnr += psnr.item()
            if save_samples and saved_count < max_samples and batch_idx == 0:
                import matplotlib.pyplot as plt
                for i in range(min(max_samples - saved_count, noisy1.size(0))):
                    noisy1_img = noisy1[i].cpu().numpy()[0]
                    noisy2_img = noisy2[i].cpu().numpy()[0]
                    denoised_img = denoised[i].cpu().numpy()[0]
                    fig, axes = plt.subplots(1,3,figsize=(12,4))
                    axes[0].imshow(noisy1_img, cmap='gray'); axes[0].axis('off'); axes[0].set_title('Noisy1')
                    axes[1].imshow(denoised_img, cmap='gray'); axes[1].axis('off'); axes[1].set_title('Denoised')
                    axes[2].imshow(noisy2_img, cmap='gray'); axes[2].axis('off'); axes[2].set_title('Target')
                    plt.tight_layout(); save_path = sample_dir / f'val_epoch_{epoch}_sample_{i}.png'; plt.savefig(save_path, dpi=120); plt.close()
                    saved_count += 1
                    if saved_count >= max_samples: break
    n = len(dataloader)
    return total_loss / n, total_psnr / n


# -------------------------------------------------------------------
# pretrained model loader
# -------------------------------------------------------------------
def load_pretrained_model(model, pretrained_path, device):
    """Load pretrained weights into model (if available)."""
    import torch

    if pretrained_path is None or not os.path.exists(pretrained_path):
        print("[WARN] No pretrained weights found, training from scratch.")
        return model

    print(f"[INFO] Loading pretrained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location=device)

    # SwinIR/NAFNet 체크포인트 호환
    if "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # key mismatch 자동 보정
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")  # DDP 제거
        new_state_dict[new_k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    print("[INFO] Pretrained weights loaded successfully.")
    return model


# --------------------------- Main train function ---------------------------

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    print(f"Using device: {device}")
    output_dir = Path(args.output_dir)
    exp_dir = output_dir / 'experiments' / args.exp_name
    checkpoint_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    sample_dir = exp_dir / 'samples'
    checkpoint_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True); sample_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)

    print("\nLoading datasets...")
    train_dataset = Noise2NoiseDataset(args.train_csv, patch_size=args.patch_size, augment=True, use_inpainting=args.use_inpainting, feather_radius=args.feather_radius)
    val_dataloader = None
    if args.val_csv:
        val_dataset = Noise2NoiseDataset(args.val_csv, patch_size=args.patch_size, augment=False, use_inpainting=args.use_inpainting, feather_radius=args.feather_radius)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    print("\nCreating model...")
    model = NAFNet(img_channel=3, width=args.width, middle_blk_num=args.middle_blk_num, enc_blk_nums=args.enc_blk_nums, dec_blk_nums=args.dec_blk_nums, use_pixelshuffle=(not args.disable_pixelshuffle)).to(device)

    if args.pretrained:
        model = load_pretrained_model(model, args.pretrained, device)

    criterion_dict = {'mse': nn.L1Loss()}
    if args.use_edge:
        print('Initializing Edge loss')
        criterion_dict['edge'] = EdgeLoss()
    if args.use_perceptual:
        print('Initializing Perceptual loss (VGG)')
        criterion_dict['perc'] = PerceptualLoss(device)
    if args.use_hf:
        print('Using high-frequency loss')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    start_epoch, best_psnr = load_checkpoint_if_exists(checkpoint_dir, model, optimizer, scheduler, resume=args.resume)

    print(f"Train samples: {len(train_dataset)} | Batch size: {args.batch_size} | Epochs: {args.epochs} (start {start_epoch})")
    print(f"Loss weights -> mse: {args.mse_weight}, edge: {args.edge_weight}, hf: {args.hf_weight}, perc: {args.perc_weight}")

    patience_counter = 0
    for epoch in range(start_epoch, args.epochs + 1):
        start_time = time.time()
        train_loss, train_mse, train_edge, train_psnr = train_epoch(model, train_dataloader, criterion_dict, optimizer, device, epoch, args)
        if val_dataloader:
            val_loss, val_psnr = validate(model, val_dataloader, criterion_dict, device, args, epoch, sample_dir)
        else:
            val_loss, val_psnr = 0, 0
        scheduler.step()
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{args.epochs} - {epoch_time:.1f}s")
        print(f" Train - Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Edge: {train_edge:.4f}) PSNR: {train_psnr:.2f} dB")
        if val_dataloader:
            print(f" Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('PSNR/train', train_psnr, epoch)
        if val_dataloader:
            writer.add_scalar('Loss/val', val_loss, epoch); writer.add_scalar('PSNR/val', val_psnr, epoch)

        # Save checkpoint
        if epoch % args.save_freq == 0:
            ckpt_path = checkpoint_dir / f'model_epoch_{epoch}.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, ckpt_path)
            print(f" Saved checkpoint: {ckpt_path}")

        current_psnr = val_psnr if val_dataloader else train_psnr
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            metrics_dict = {'train_loss': train_loss, 'train_mse': train_mse, 'train_edge': train_edge, 'val_loss': val_loss, 'val_psnr': val_psnr}
            best_path = checkpoint_dir / f'best_model_epoch_{epoch}_psnr_{best_psnr:.2f}.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_psnr': best_psnr, 'metrics': metrics_dict}, best_path)
            latest_best = checkpoint_dir / 'best_model.pth'
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_psnr': best_psnr, 'metrics': metrics_dict}, latest_best)
            print(f" ⭐ New best model! PSNR: {best_psnr:.2f} dB (Epoch {epoch}) Saved: {best_path.name}")
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    writer.close()
    print('Training completed!')
    print(f' Best PSNR: {best_psnr:.2f} dB')

# --------------------------- Argument parser ---------------------------

def main():
    parser = argparse.ArgumentParser(description='Improved NAFNet Training')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='Outputs')
    parser.add_argument('--exp_name', type=str, default='nafnet_pixelshuffle_improved')
    parser.add_argument('--use_inpainting', action='store_true', default=False)
    parser.add_argument('--feather_radius', type=float, default=7.0)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--middle_blk_num', type=int, default=8)
    parser.add_argument('--enc_blk_nums', type=int, nargs='+', default=[2,2,4,8])
    parser.add_argument('--dec_blk_nums', type=int, nargs='+', default=[2,2,2,2])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mse_weight', type=float, default=0.8)
    parser.add_argument('--edge_weight', type=float, default=0.2)
    parser.add_argument('--hf_weight', type=float, default=0.1)
    parser.add_argument('--perc_weight', type=float, default=0.0)
    parser.add_argument('--use_edge', action='store_true', default=True)
    parser.add_argument('--use_perceptual', action='store_true', default=False)
    parser.add_argument('--use_hf', action='store_true', default=False)
    parser.add_argument('--disable_pixelshuffle', action='store_true', default=False)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
