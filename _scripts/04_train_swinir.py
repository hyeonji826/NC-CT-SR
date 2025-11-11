# -*- coding: utf-8 -*-
"""
SwinIR Fine-tuning for CT Super-Resolution
NC (low-res) -> CE (high-res)
With comprehensive evaluation metrics and visualization
"""
import os
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
from tqdm import tqdm
os.environ["MPLBACKEND"] = "Agg"   # 반드시 pyplot 임포트 전에
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# -------------------- SwinIR repo path resolve --------------------
script_dir = Path(__file__).parent
possible_paths = [
    script_dir.parent / "SwinIR",          # E:\LD-CT SR\SwinIR
    script_dir.parent.parent / "SwinIR",
    Path("SwinIR"),
]
SWINIR_PATH = None
for path in possible_paths:
    if path.exists() and (path / "models" / "network_swinir.py").exists():
        SWINIR_PATH = path
        break
if SWINIR_PATH is None:
    print("ERROR: Cannot find SwinIR repository.")
    for p in possible_paths: print(f"  - {p.absolute()}")
    print("Example:\n  cd 'E:\\LD-CT SR'\n  git clone https://github.com/JingyunLiang/SwinIR.git")
    sys.exit(1)
sys.path.insert(0, str(SWINIR_PATH))
print(f"Found SwinIR at: {SWINIR_PATH.absolute()}")

try:
    from models.network_swinir import SwinIR
except ImportError as e:
    print(f"ERROR: Cannot import SwinIR: {e}")
    sys.exit(1)

# ==================== METRICS ====================
def _to_np2d_batch(x: torch.Tensor):
    """
    Accepts (B,1,H,W) or (1,H,W) or (H,W); returns list of 2D numpy arrays.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
    arr = x.numpy()
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:  # (1,H,W) or (C,H,W)
        return [arr[0]]
    if arr.ndim == 4:  # (B,1,H,W) or (B,C,H,W)
        return [arr[i, 0] for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported tensor shape for metrics: {arr.shape}")

def calculate_psnr(sr, hr, data_range=1.0):
    sr_list = _to_np2d_batch(sr)
    hr_list = _to_np2d_batch(hr)
    vals = [psnr(h, s, data_range=data_range) for s, h in zip(sr_list, hr_list)]
    return float(np.mean(vals))

def calculate_ssim(sr, hr, data_range=1.0):
    sr_list = _to_np2d_batch(sr)
    hr_list = _to_np2d_batch(hr)
    vals = []
    for s, h in zip(sr_list, hr_list):
        if min(s.shape) < 7 or min(h.shape) < 7:
            vals.append(0.0); continue
        try:
            vals.append(ssim(h, s, data_range=data_range, win_size=7))
        except Exception:
            vals.append(0.0)
    return float(np.mean(vals))

def calculate_mae(sr, hr):
    return torch.mean(torch.abs(sr - hr)).item()

def calculate_rmse(sr, hr):
    return torch.sqrt(torch.mean((sr - hr) ** 2)).item()

# ==================== DATASET ====================
class CTSuperResDataset(Dataset):
    """CT Super-Resolution Dataset - Slice-level pairing"""

    def __init__(self, slice_pairs_csv, root_dir, patch_size=64, augment=True):
        self.root = Path(root_dir)
        self.patch_size = int(patch_size)
        self.augment = bool(augment)

        # CSV 경로 안전 처리 (절대면 그대로, 상대면 root 기준)
        slice_pairs_csv = Path(slice_pairs_csv)
        csv_path = slice_pairs_csv if slice_pairs_csv.is_absolute() else (self.root / slice_pairs_csv)
        df = pd.read_csv(csv_path)

        self.slice_pairs = []
        for _, row in df.iterrows():
            nc_path = Path(row['nc_path'])
            ce_path = Path(row['ce_path'])
            # 데이터셋 CSV에 절대경로가 들어있다면 그대로 사용, 아니면 root 기준으로 보정
            if not nc_path.is_absolute():
                nc_path = (self.root / nc_path).resolve()
            if not ce_path.is_absolute():
                ce_path = (self.root / ce_path).resolve()
            if nc_path.exists() and ce_path.exists():
                self.slice_pairs.append({
                    'nc_path': nc_path,
                    'ce_path': ce_path,
                    'nc_slice': int(row['nc_slice']),
                    'ce_slice': int(row['ce_slice']),
                    'case_id' : row.get('case_id', None)
                })
        print(f"[Dataset] Loaded {len(self.slice_pairs)} slice pairs from {csv_path}")

        # 캐시
        self.volume_cache = {}

    def __len__(self):
        return len(self.slice_pairs)

    def load_volume(self, path: Path):
        key = str(path)
        if key not in self.volume_cache:
            img = sitk.ReadImage(key)
            arr = sitk.GetArrayFromImage(img).astype(np.float32)
            # 정규화: [0,1] 가정 (이미 HU→clip→norm을 전처리에서 했다면 주석 처리 가능)
            vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
            if vmax > vmin:
                arr = (arr - vmin) / (vmax - vmin)
            arr = np.clip(arr, 0.0, 1.0)
            self.volume_cache[key] = arr
        return self.volume_cache[key]

    def extract_corresponding_patches(self, nc_slice, ce_slice, patch_size):
        H_nc, W_nc = nc_slice.shape
        H_ce, W_ce = ce_slice.shape

        # 무작위 패치 시작점 (NC 기준)
        h_start_nc = random.randint(0, max(0, H_nc - patch_size))
        w_start_nc = random.randint(0, max(0, W_nc - patch_size))
        nc_patch = nc_slice[h_start_nc:h_start_nc+patch_size, w_start_nc:w_start_nc+patch_size]

        # 해상도 비율로 CE의 대응 위치 계산 (동일 크기이면 ratio=1)
        ratio_h = H_ce / max(1, H_nc)
        ratio_w = W_ce / max(1, W_nc)
        h_start_ce = int(h_start_nc * ratio_h)
        w_start_ce = int(w_start_nc * ratio_w)
        h_start_ce = min(h_start_ce, max(0, H_ce - patch_size))
        w_start_ce = min(w_start_ce, max(0, W_ce - patch_size))
        ce_patch = ce_slice[h_start_ce:h_start_ce+patch_size, w_start_ce:w_start_ce+patch_size]

        # 패딩
        if nc_patch.shape != (patch_size, patch_size):
            pad = np.zeros((patch_size, patch_size), dtype=np.float32)
            pad[:nc_patch.shape[0], :nc_patch.shape[1]] = nc_patch
            nc_patch = pad
        if ce_patch.shape != (patch_size, patch_size):
            pad = np.zeros((patch_size, patch_size), dtype=np.float32)
            pad[:ce_patch.shape[0], :ce_patch.shape[1]] = ce_patch
            ce_patch = pad
        return nc_patch, ce_patch

    def augment_patch(self, lr_patch, hr_patch):
        if random.random() > 0.5:
            lr_patch = np.fliplr(lr_patch).copy()
            hr_patch = np.fliplr(hr_patch).copy()
        if random.random() > 0.5:
            lr_patch = np.flipud(lr_patch).copy()
            hr_patch = np.flipud(hr_patch).copy()
        k = random.randint(0, 3)
        if k:
            lr_patch = np.rot90(lr_patch, k).copy()
            hr_patch = np.rot90(hr_patch, k).copy()
        return lr_patch, hr_patch

    def __getitem__(self, idx):
        pair = self.slice_pairs[idx]
        nc_vol = self.load_volume(pair['nc_path'])
        ce_vol = self.load_volume(pair['ce_path'])

        nc_slice = nc_vol[pair['nc_slice']]
        ce_slice = ce_vol[pair['ce_slice']]

        lr_patch, hr_patch = self.extract_corresponding_patches(
            nc_slice, ce_slice, self.patch_size
        )
        if self.augment:
            lr_patch, hr_patch = self.augment_patch(lr_patch, hr_patch)

        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)  # (1,H,W)
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)  # (1,H,W)
        return lr_tensor, hr_tensor

# ==================== MODEL ====================
def load_pretrained_swinir(weight_path, upscale=1, in_chans=1):
    if upscale == 1:
        model = SwinIR(
            upscale=1, in_chans=in_chans, img_size=64, window_size=8, img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
            upsampler='', resi_connection='1conv'
        )
    else:
        model = SwinIR(
            upscale=upscale, in_chans=in_chans, img_size=64, window_size=8, img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
            upsampler='pixelshuffle', resi_connection='1conv'
        )

    weight_path = Path(weight_path)
    if weight_path.exists():
        print(f"Loading pretrained weights from: {weight_path}")
        pretrained_dict = torch.load(weight_path, map_location='cpu')
        if 'params' in pretrained_dict: pretrained_dict = pretrained_dict['params']
        elif 'params_ema' in pretrained_dict: pretrained_dict = pretrained_dict['params_ema']

        model_dict = model.state_dict()

        # RGB->Gray 변환
        if in_chans == 1:
            if 'conv_first.weight' in pretrained_dict:
                w = pretrained_dict['conv_first.weight']          # (C_out,3,H,W)
                pretrained_dict['conv_first.weight'] = w.mean(dim=1, keepdim=True)
            if 'conv_last.weight' in pretrained_dict:
                w = pretrained_dict['conv_last.weight']           # (3,C_in,H,W)
                pretrained_dict['conv_last.weight'] = w.mean(dim=0, keepdim=True)
            if 'conv_last.bias' in pretrained_dict:
                b = pretrained_dict['conv_last.bias']             # (3,)
                pretrained_dict['conv_last.bias'] = b.mean(dim=0, keepdim=True)

        # Key/shape 일치하는 항목만 로드
        filtered = {k: v for k, v in pretrained_dict.items()
                    if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(filtered)}/{len(model_dict)} layers from pretrained model")
    else:
        print(f"WARNING: Pretrained weight not found at {weight_path}. Training from scratch.")
    return model

# ==================== VISUALIZATION ====================
def save_comparison_image(lr, sr, hr, save_path, epoch):
    lr_np = lr[0, 0].detach().cpu().numpy()
    sr_np = sr[0, 0].detach().cpu().numpy()
    hr_np = hr[0, 0].detach().cpu().numpy()

    error_sr = np.abs(sr_np - hr_np)
    error_lr = np.abs(lr_np - hr_np)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(lr_np, cmap='gray', vmin=0, vmax=1); axes[0, 0].set_title('Input (NC)'); axes[0, 0].axis('off')
    axes[0, 1].imshow(sr_np, cmap='gray', vmin=0, vmax=1); axes[0, 1].set_title('Output (SR)'); axes[0, 1].axis('off')
    axes[0, 2].imshow(hr_np, cmap='gray', vmin=0, vmax=1); axes[0, 2].set_title('Target (CE)'); axes[0, 2].axis('off')

    im1 = axes[1, 0].imshow(error_lr, cmap='hot', vmin=0, vmax=0.3); axes[1, 0].set_title('LR Error'); axes[1, 0].axis('off'); plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    im2 = axes[1, 1].imshow(error_sr, cmap='hot', vmin=0, vmax=0.3); axes[1, 1].set_title('SR Error'); axes[1, 1].axis('off'); plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    improvement = error_lr - error_sr
    im3 = axes[1, 2].imshow(improvement, cmap='RdYlGn', vmin=-0.1, vmax=0.1); axes[1, 2].set_title('Improvement (G=Better)'); axes[1, 2].axis('off'); plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    plt.suptitle(f'Epoch {epoch} - Visual Comparison', fontsize=16)
    plt.tight_layout()
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

def plot_metrics_curve(history, save_path):
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].plot(epochs, history['train_loss'], label='Train'); axes[0, 0].plot(epochs, history['val_loss'], label='Val'); axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)
    axes[0, 1].plot(epochs, history['val_psnr']); axes[0, 1].set_title('PSNR (dB)'); axes[0, 1].grid(True)
    axes[0, 2].plot(epochs, history['val_ssim']); axes[0, 2].set_title('SSIM'); axes[0, 2].grid(True)
    axes[1, 0].plot(epochs, history['val_mae']); axes[1, 0].set_title('MAE'); axes[1, 0].grid(True)
    axes[1, 1].plot(epochs, history['val_rmse']); axes[1, 1].set_title('RMSE'); axes[1, 1].grid(True)
    axes[1, 2].plot(epochs, history['lr']); axes[1, 2].set_title('LR (log)'); axes[1, 2].set_yscale('log'); axes[1, 2].grid(True)
    plt.tight_layout()
    save_path = Path(save_path); save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()

# ==================== TRAINING ====================
class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- Root & dirs ----
        self.root = Path(args.root).resolve()              # <== FIX: define self.root
        self.exp_dir = (self.root / args.exp_dir).resolve()
        self.ckpt_dir = self.exp_dir / 'checkpoints'
        self.log_dir  = self.exp_dir / 'logs'
        self.vis_dir  = self.exp_dir / 'visualizations'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # ---- Load all slice pairs and split by case ----
        print("\n" + "="*80)
        print("Loading datasets...")
        csv_path = (self.root / args.slice_pairs_csv).resolve()
        df_slices = pd.read_csv(csv_path)

        if 'case_id' not in df_slices.columns:
            # case_id가 없다면 파일명/경로에서 id를 추정해 생성하도록 필요시 수정
            raise ValueError("slice_pairs CSV에 'case_id' 컬럼이 필요합니다.")

        unique_cases = df_slices['case_id'].unique()
        n_cases = len(unique_cases)
        n_train_cases = max(1, int(n_cases * 0.8))

        rng = np.random.default_rng(42)
        train_cases = set(rng.choice(unique_cases, n_train_cases, replace=False))

        train_slices = df_slices[df_slices['case_id'].isin(train_cases)].reset_index(drop=True)
        val_slices   = df_slices[~df_slices['case_id'].isin(train_cases)].reset_index(drop=True)

        print(f"Case-level split:")
        print(f"  Train cases: {len(train_cases)}, slices: {len(train_slices)}")
        print(f"  Val cases: {n_cases - len(train_cases)}, slices: {len(val_slices)}")

        train_csv = self.exp_dir / 'train_slices.csv'
        val_csv   = self.exp_dir / 'val_slices.csv'
        train_slices.to_csv(train_csv, index=False)
        val_slices.to_csv(val_csv, index=False)

        self.train_dataset = CTSuperResDataset(
            slice_pairs_csv=train_csv, root_dir=self.root,
            patch_size=args.patch_size, augment=True
        )
        self.val_dataset = CTSuperResDataset(
            slice_pairs_csv=val_csv, root_dir=self.root,
            patch_size=args.patch_size, augment=False
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True
        )
        print(f"Train: {len(self.train_dataset)} samples, Val: {len(self.val_dataset)} samples")

        # ---- Model ----
        print("\nInitializing model...")
        pretrained_path = (self.root / args.pretrained_model).resolve()
        self.model = load_pretrained_swinir(
            pretrained_path, upscale=args.upscale, in_chans=1
        ).to(self.device)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs: "
                  f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-7)
        self.criterion = nn.L1Loss()
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_psnr = 0.0

        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'val_psnr': [], 'val_ssim': [],
            'val_mae': [], 'val_rmse': [], 'lr': []
        }
        print("="*80 + "\n")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.args.epochs} [TRAIN]")
        for lr, hr in pbar:
            lr, hr = lr.to(self.device), hr.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            sr = self.model(lr)
            loss = self.criterion(sr, hr)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        return total_loss / max(1, len(self.train_loader))

    def validate(self, epoch):
        self.model.eval()
        total_loss = total_psnr = total_ssim = total_mae = total_rmse = 0.0
        save_first_batch = True
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.args.epochs} [VAL]")
            for lr, hr in pbar:
                lr, hr = lr.to(self.device), hr.to(self.device)
                sr = self.model(lr)
                loss = self.criterion(sr, hr)

                batch_psnr = calculate_psnr(sr, hr)
                batch_ssim = calculate_ssim(sr, hr)
                batch_mae  = calculate_mae(sr, hr)
                batch_rmse = calculate_rmse(sr, hr)

                total_loss += loss.item()
                total_psnr += batch_psnr
                total_ssim += batch_ssim
                total_mae  += batch_mae
                total_rmse += batch_rmse

                pbar.set_postfix({'loss': f'{loss.item():.4f}',
                                  'psnr': f'{batch_psnr:.2f}dB',
                                  'ssim': f'{batch_ssim:.4f}'})

                if save_first_batch:
                    vis_path = self.vis_dir / f'epoch_{epoch:04d}.png'
                    save_comparison_image(lr, sr, hr, vis_path, epoch)
                    save_first_batch = False

        n = max(1, len(self.val_loader))
        return {
            'loss': total_loss / n,
            'psnr': total_psnr / n,
            'ssim': total_ssim / n,
            'mae' : total_mae  / n,
            'rmse': total_rmse / n
        }

    def save_checkpoint(self, epoch, is_best=False):
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_psnr': self.best_psnr,
            'metrics_history': self.metrics_history,
        }
        latest_path = self.ckpt_dir / 'latest.pth'
        torch.save(ckpt, latest_path)
        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(ckpt, best_path)
        if epoch % self.args.save_freq == 0:
            epoch_path = self.ckpt_dir / f'epoch_{epoch:04d}.pth'
            torch.save(ckpt, epoch_path)

    def train(self):
        print("Starting training...\n")
        print("="*100)
        print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'PSNR (dB)':<12} {'SSIM':<12} {'MAE':<12} {'RMSE':<12} {'LR':<12}")
        print("="*100)
        for epoch in range(self.start_epoch + 1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()
            lr_val = self.optimizer.param_groups[0]['lr']

            print(f"{epoch:<8} {train_loss:<12.4f} {val_metrics['loss']:<12.4f} "
                  f"{val_metrics['psnr']:<12.2f} {val_metrics['ssim']:<12.4f} "
                  f"{val_metrics['mae']:<12.4f} {val_metrics['rmse']:<12.4f} {lr_val:<12.2e}")

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val',   val_metrics['loss'], epoch)
            self.writer.add_scalar('Metrics/PSNR', val_metrics['psnr'], epoch)
            self.writer.add_scalar('Metrics/SSIM', val_metrics['ssim'], epoch)
            self.writer.add_scalar('Metrics/MAE',  val_metrics['mae'], epoch)
            self.writer.add_scalar('Metrics/RMSE', val_metrics['rmse'], epoch)
            self.writer.add_scalar('LR', lr_val, epoch)

            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['val_psnr'].append(val_metrics['psnr'])
            self.metrics_history['val_ssim'].append(val_metrics['ssim'])
            self.metrics_history['val_mae'].append(val_metrics['mae'])
            self.metrics_history['val_rmse'].append(val_metrics['rmse'])
            self.metrics_history['lr'].append(lr_val)

            plot_metrics_curve(self.metrics_history, self.vis_dir / 'training_curves.png')

            is_best = val_metrics['psnr'] > self.best_psnr
            if is_best:
                self.best_psnr = val_metrics['psnr']
                self.best_val_loss = val_metrics['loss']
                print(f"  → ⭐ New best model! PSNR: {self.best_psnr:.2f}dB")
            self.save_checkpoint(epoch, is_best)

        print("="*100)
        print(f"\n🎉 Training complete!")
        print(f"Best PSNR: {self.best_psnr:.2f}dB")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints: {self.ckpt_dir}")
        print(f"Visualizations: {self.vis_dir}")
        print(f"Tensorboard logs: {self.log_dir}")
        print("="*100)
        self.writer.close()

# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='SwinIR Fine-tuning for CT SR')
    # Data
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--slice-pairs-csv', default='Data/slice_pairs.csv')
    parser.add_argument('--exp-dir', default='Outputs/experiments/swinir_x1_slicepairs')
    # Model
    parser.add_argument('--pretrained-model', default=r'Weights/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth')
    parser.add_argument('--upscale', type=int, default=1, help='1: enhancement only, 2/4: SR')
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-freq', type=int, default=10)
    args = parser.parse_args()

    # GPU info
    if torch.cuda.is_available():
        print(f"\n{'='*80}\nGPU Configuration:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"{'='*80}\n")
    else:
        print("WARNING: CUDA not available, using CPU")

    # Seeds
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
