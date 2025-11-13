#!/usr/bin/env python3
"""
NAFNet Noise2Noise Training Script
Self-supervised denoising using Noise2Noise pairs
Stage 1: NC denoising (구조 보존하며 노이즈만 제거)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd
import time
import json
from torch.utils.tensorboard import SummaryWriter


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
    """NAFNet 모델"""
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
    def __init__(self, csv_path, patch_size=256, augment=True):
        self.csv_path = Path(csv_path)
        self.pairs_df = pd.read_csv(csv_path)
        self.patch_size = patch_size
        self.augment = augment
        
        print(f"Loaded {len(self.pairs_df)} Noise2Noise pairs from {csv_path}")
    
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
    """사전학습 모델 로드"""
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
    
    # 키 이름 매칭 (필요시)
    model_dict = model.state_dict()
    pretrained_dict = {}
    
    for k, v in state_dict.items():
        # 'module.' 제거
        new_k = k.replace('module.', '')
        
        # 모델에 해당 키가 있고 shape이 일치하면 로드
        if new_k in model_dict and model_dict[new_k].shape == v.shape:
            pretrained_dict[new_k] = v
        else:
            print(f"  [SKIP] {new_k}: shape mismatch or not found")
    
    # 로드된 파라미터 통계
    loaded_keys = len(pretrained_dict)
    total_keys = len(model_dict)
    print(f"\nLoaded {loaded_keys}/{total_keys} parameters from pretrained model")
    print(f"Loading ratio: {loaded_keys/total_keys*100:.1f}%")
    
    # 파라미터 로드
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    print(f"{'='*80}\n")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """1 epoch 학습 (Noise2Noise)"""
    model.train()
    total_loss = 0
    total_psnr = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for noisy1, noisy2 in pbar:
        noisy1, noisy2 = noisy1.to(device), noisy2.to(device)
        
        # Forward: noisy1 → denoise → compare with noisy2
        denoised = model(noisy1)
        loss = criterion(denoised, noisy2)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            psnr = calculate_psnr(denoised, noisy2)
        
        total_loss += loss.item()
        total_psnr += psnr.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{psnr.item():.2f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / len(dataloader)
    
    return avg_loss, avg_psnr


def validate(model, dataloader, criterion, device, epoch=None, sample_dir=None):
    """Validation (Noise2Noise)"""
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
            loss = criterion(denoised, noisy2)
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
        augment=True
    )
    
    # Validation dataset (if provided)
    val_dataloader = None
    if args.val_csv:
        val_dataset = Noise2NoiseDataset(
            args.val_csv,
            patch_size=args.patch_size,
            augment=False
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
    
    # Loss & Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 학습 시작
    print(f"\n{'='*80}")
    print("Starting Noise2Noise Training (Self-supervised Denoising)")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_dataset)}")
    if val_dataloader:
        print(f"Val samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Strategy: Self-supervised (Noise2Noise)")
    print(f"Goal: Denoise NC-CT while preserving structure")
    print(f"{'='*80}\n")
    
    best_psnr = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_psnr = train_epoch(
            model, train_dataloader, criterion, optimizer, device, epoch
        )
        
        # Validate
        if val_dataloader:
            val_loss, val_psnr = validate(model, val_dataloader, criterion, device, epoch, sample_dir)
        else:
            val_loss, val_psnr = 0, 0
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}/{args.epochs} - {epoch_time:.1f}s")
        print(f"  Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f} dB")
        if val_dataloader:
            print(f"  Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f} dB")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
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
                'best_psnr': best_psnr,
                'config': {
                    'width': args.width,
                    'middle_blk_num': args.middle_blk_num,
                    'enc_blk_nums': args.enc_blk_nums,
                    'dec_blk_nums': args.dec_blk_nums
                }
            }, latest_best)
            
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
    parser = argparse.ArgumentParser(description='NAFNet Noise2Noise Training (Self-supervised Denoising)')
    
    # Data
    parser.add_argument('--train_csv', type=str, required=True,
                       help='Path to training Noise2Noise pairs CSV')
    parser.add_argument('--val_csv', type=str, default=None,
                       help='Path to validation Noise2Noise pairs CSV (optional)')
    parser.add_argument('--output_dir', type=str, default='Outputs',
                       help='Output directory for experiments')
    parser.add_argument('--exp_name', type=str, default='nafnet_noise2noise',
                       help='Experiment name (creates subdirectory in output_dir)')
    
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
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
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