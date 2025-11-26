# train_n2n.py - Self-Supervised Training with Neighbor2Neighbor + Wavelet

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import argparse
import numpy as np
import nibabel as nib

# SwinIR path
sys.path.insert(0, r'E:\LD-CT SR\_externals\SwinIR')
from models.network_swinir import SwinIR

from dataset_n2n import NCCTDenoiseDataset
from losses_n2n import CombinedN2NWaveletLoss, Neighbor2NeighborLoss
from utils import (
    save_checkpoint, load_checkpoint, save_sample_images,
    cleanup_old_checkpoints, EarlyStopping, WarmupScheduler
)

# -------------------------------------------------------------------------
# Fixed full-slice sampler (HN/LN 한 쌍 선택)
# -------------------------------------------------------------------------
def load_fixed_full_slices(nc_ct_dir, hu_window, seed=None):
    from scipy import ndimage
    import random

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    nc_ct_dir = Path(nc_ct_dir)
    files = sorted(list(nc_ct_dir.glob("*.nii.gz")))
    candidates = []

    print(f"\n🔍 Analyzing slices (RELATIVE noise-based selection, seed={seed})...")
    max_files = min(50, len(files))

    for file_idx, file_path in enumerate(files[:max_files]):
        try:
            nii = nib.load(str(file_path))
            volume = nii.get_fdata()
            D = volume.shape[2]

            slice_indices = [D//5, D//3, D//2, 2*D//3, 4*D//5]
            for slice_idx in slice_indices:
                if slice_idx >= D:
                    continue

                slice_2d = volume[:, :, slice_idx]
                h, w = slice_2d.shape

                center_ratio_w = 0.55
                center_ratio_h = 0.60
                margin_h = int(h * (1 - center_ratio_h) / 2)
                margin_w = int(w * (1 - center_ratio_w) / 2)

                center_slice = slice_2d[margin_h:h - margin_h,
                                        margin_w:w - margin_w]

                tissue_mask = (center_slice > -100) & (center_slice < 100)
                if tissue_mask.sum() < 1000:
                    continue

                noise_std = center_slice[tissue_mask].std()

                edge_x = ndimage.sobel(center_slice, axis=0)
                edge_y = ndimage.sobel(center_slice, axis=1)
                edge_mag = np.sqrt(edge_x**2 + edge_y**2)
                edge_score = edge_mag[tissue_mask].mean()

                candidates.append({
                    'file_path': file_path,
                    'slice_idx': slice_idx,
                    'noise_std': noise_std,
                    'edge_score': edge_score,
                    'slice_2d': center_slice
                })
        except Exception:
            continue

        if file_idx >= 49:
            break

    if len(candidates) < 2:
        raise ValueError(f"Not enough valid slices found (only {len(candidates)})")

    noise_values = [c['noise_std'] for c in candidates]
    p10 = np.percentile(noise_values, 10)
    p50 = np.percentile(noise_values, 50)
    p90 = np.percentile(noise_values, 90)

    print(f"  Scanned {max_files}/{len(files)} files...")
    print(f"   Noise distribution: 10th={p10:.1f}, median={p50:.1f}, 90th={p90:.1f} HU")

    hn_candidates = [c for c in candidates if c['noise_std'] >= p90]
    if len(hn_candidates) == 0:
        hn_candidates = sorted(candidates, key=lambda x: x['noise_std'], reverse=True)[:10]
    high_noise = max(hn_candidates, key=lambda x: x['noise_std'])

    ln_candidates = [c for c in candidates if c['noise_std'] <= p10]
    if len(ln_candidates) < 5:
        ln_candidates = sorted(candidates, key=lambda x: x['noise_std'])[:max(5, len(candidates)//7)]
    low_noise = max(ln_candidates, key=lambda x: x['edge_score'])

    hn_pct = (sum(1 for c in candidates if c['noise_std'] > high_noise['noise_std']) / len(candidates)) * 100
    ln_pct = (sum(1 for c in candidates if c['noise_std'] > low_noise['noise_std']) / len(candidates)) * 100

    print(f"✅ Selected 2 representative slices (RELATIVE extremes, seed={seed}):")
    print(f"   [HN] {high_noise['file_path'].name} slice {high_noise['slice_idx']}  "
          f"Noise: {high_noise['noise_std']:.1f} HU (top {hn_pct:.1f}%)  "
          f"Edge: {high_noise['edge_score']:.1f}")
    print(f"   [LN] {low_noise['file_path'].name} slice {low_noise['slice_idx']}  "
          f"Noise: {low_noise['noise_std']:.1f} HU (bottom {ln_pct:.1f}%)  "
          f"Edge: {low_noise['edge_score']:.1f}")
    print(f"   Noise ratio (HN/LN): {high_noise['noise_std'] / max(low_noise['noise_std'], 1e-6):.2f}x")
    print(f"   ⚠️  Note: Entire dataset is noisy (10th %ile = {p10:.1f} HU)")

    slices = []
    slice_info = []

    for label, cand in [('HN', high_noise), ('LN', low_noise)]:
        slice_2d = cand['slice_2d']
        slice_2d = np.clip(slice_2d, hu_window[0], hu_window[1])
        slice_2d = (slice_2d - hu_window[0]) / (hu_window[1] - hu_window[0])
        slice_2d = slice_2d.astype(np.float32)

        slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)
        slices.append(slice_tensor)

        slice_info.append({
            'label': label,
            'noise_std_hu': cand['noise_std'],
            'edge_score': cand.get('edge_score', 0.0),
            'file': cand['file_path'].name,
            'slice_idx': cand['slice_idx']
        })

    return slices, slice_info

# -------------------------------------------------------------------------
# Config / arg parsing
# -------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='_scripts_4_wavelet/config/config_n2n.yaml')
    parser.add_argument('--exp', type=str, default='debug')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# -------------------------------------------------------------------------
# Main training
# -------------------------------------------------------------------------
def train_n2n():
    args = parse_args()

    script_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"\n📄 Using config file: {config_path}")
    config = load_config(config_path)
    exp_name = args.exp  # 현재는 사용 안 하지만 남겨둠

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"   GPU : {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Output dirs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config['data']['output_dir']) / f'n2n_wavelet_{timestamp}'
    ckpt_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    sample_dir = exp_dir / 'samples'
    for d in [ckpt_dir, log_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    writer = SummaryWriter(log_dir)
    print(f"\nTensorBoard: tensorboard --logdir={log_dir}")

    # Dataset
    print("\n📂 Loading NC-CT dataset (self-supervised)...")
    full_dataset = NCCTDenoiseDataset(
        nc_ct_dir=config['data']['nc_ct_dir'],
        hu_window=config['preprocessing']['hu_window'],
        patch_size=config['preprocessing']['patch_size'],
        config_aug=config['training']['augmentation'],
        mode='train'
    )

    val_size = int(len(full_dataset) * config['training']['val_split'])
    test_size = int(len(full_dataset) * config['training'].get('test_split', 0.1))
    train_size = len(full_dataset) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        persistent_workers=config['training']['num_workers'] > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples  : {len(val_dataset)}")
    print(f"   Test samples : {len(test_dataset)}")
    print(f"   Batches/epoch: {len(train_loader)}")

    # Save test set file list
    test_indices_path = exp_dir / 'test_indices.txt'
    with open(test_indices_path, 'w') as f:
        for idx in test_dataset.indices:
            file_idx = idx % len(full_dataset.files)
            f.write(f"{full_dataset.files[file_idx]}\n")
    print(f"   Test set file list saved: {test_indices_path}")

    # Model
    print("\n🧱 Building SwinIR model...")
    model = SwinIR(
        upscale=config['model']['upscale'],
        in_chans=config['model']['in_chans'],
        img_size=config['model']['img_size'],
        window_size=config['model']['window_size'],
        img_range=config['model']['img_range'],
        depths=config['model']['depths'],
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        upsampler=config['model']['upsampler'],
        resi_connection=config['model']['resi_connection']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters   : {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Pretrained weights
    pretrained_path = config['model'].get('pretrained')
    if pretrained_path and Path(pretrained_path).exists():
        print(f"\nLoading pretrained weights:\n   {pretrained_path}")
        try:
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if 'params' in pretrained_dict:
                pretrained_dict = pretrained_dict['params']
            elif 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']

            model_dict = model.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"   Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
        except Exception as e:
            print(f"   Failed to load pretrained: {e}")

    # Loss
    print("\n🔧 Setting up loss function...")
    loss_type = config['training'].get('loss_type', 'n2n_wavelet_edge')

    if loss_type in ['n2n_wavelet', 'n2n_wavelet_edge']:
        criterion = CombinedN2NWaveletLoss(
            n2n_gamma=config['training']['n2n_gamma'],
            wavelet_weight=config['training']['wavelet_weight'],
            wavelet_threshold=config['training']['wavelet_threshold'],
            wavelet_levels=config['training']['wavelet_levels'],
            hu_window=tuple(config['preprocessing']['hu_window']),
            adaptive=True,
            target_noise=config['training'].get('target_noise', 0.15),
            adaptive_weight_range=tuple(config['training'].get('adaptive_weight_range', [0.3, 3.0])),
            edge_weight=config['training'].get('edge_weight', 0.05),
        ).to(device)
    else:
        criterion = Neighbor2NeighborLoss(
            gamma=config['training']['n2n_gamma']
        ).to(device)
        print("⚠️ Using basic N2N loss (no wavelet/edge)")

    # Optimizer / scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['eta_min']
    )

    warmup = WarmupScheduler(
        optimizer,
        warmup_epochs=config['training']['warmup_epochs'],
        warmup_lr=config['training']['warmup_lr'],
        base_lr=config['training']['learning_rate']
    )

    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        min_delta=config['training']['early_stopping']['min_delta']
    )

    use_amp = config['training']['use_amp']
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    # Resume
    start_epoch = 1
    best_val_loss = float('inf')
    if config['training']['resume']:
        resume_path = Path(config['training']['resume'])
        if resume_path.exists():
            start_epoch, loaded_val_loss = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch += 1
            best_val_loss = loaded_val_loss
            print(f"\nResumed from epoch {start_epoch-1}, best val loss {best_val_loss:.4f}")
        else:
            print(f"\nWARNING: resume path not found: {resume_path}")

    print("\n==================== Start Training ====================\n")
    global_step = 0
    balance_warnings = 0
    target_balance = config['monitoring'].get('target_balance_ratio', 20.0)

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        model.train()
        train_losses = []
        train_loss_details = {k: [] for k in
                              ['n2n_rec', 'n2n_reg', 'n2n_total',
                               'wavelet_weighted', 'total',
                               'balance_ratio', 'estimated_noise', 'adaptive_weight']}

        if warmup.is_warmup():
            warmup.step()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['num_epochs']}")

        for noisy_batch in pbar:
            noisy = noisy_batch.to(device, non_blocking=True)
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    loss, loss_dict = criterion(model, noisy)
                scaler.scale(loss).backward()
                if config['training']['gradient_clip'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['gradient_clip']
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_dict = criterion(model, noisy)
                loss.backward()
                if config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['gradient_clip']
                    )
                optimizer.step()

            train_losses.append(loss.item())
            for k in train_loss_details.keys():
                if k in loss_dict:
                    train_loss_details[k].append(loss_dict[k])

            global_step += 1
            writer.add_scalar('Train/total_loss', loss.item(), global_step)
            if 'n2n_total' in loss_dict:
                writer.add_scalar('Train/n2n_total', loss_dict['n2n_total'], global_step)
            if 'wavelet_weighted' in loss_dict:
                writer.add_scalar('Train/wavelet', loss_dict['wavelet_weighted'], global_step)
            if 'balance_ratio' in loss_dict:
                writer.add_scalar('Train/balance_ratio', loss_dict['balance_ratio'], global_step)
            if 'estimated_noise' in loss_dict:
                writer.add_scalar('Train/estimated_noise', loss_dict['estimated_noise'], global_step)
            if 'adaptive_weight' in loss_dict:
                writer.add_scalar('Train/adaptive_weight', loss_dict['adaptive_weight'], global_step)

            pbar_dict = {'loss': f"{loss.item():.4f}",
                         'lr': f"{optimizer.param_groups[0]['lr']:.6f}"}
            if 'n2n_total' in loss_dict:
                pbar_dict['n2n'] = f"{loss_dict['n2n_total']:.4f}"
            if 'wavelet_weighted' in loss_dict:
                pbar_dict['wav'] = f"{loss_dict['wavelet_weighted']:.4f}"
            if 'estimated_noise' in loss_dict and loss_dict['estimated_noise'] > 0:
                pbar_dict['noise'] = f"{loss_dict['estimated_noise']*400:.1f}"
            pbar.set_postfix(pbar_dict)

        if not warmup.is_warmup():
            scheduler.step()

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_details = {k: (sum(v)/len(v) if v else 0.0)
                             for k, v in train_loss_details.items()}

        # ---------------- Validation ----------------
        model.eval()
        val_losses = []
        val_loss_details = {k: [] for k in
                            ['n2n_total', 'wavelet_weighted',
                             'balance_ratio', 'estimated_noise', 'adaptive_weight']}

        with torch.no_grad():
            for noisy_batch in tqdm(val_loader, desc="Validation", leave=False):
                noisy = noisy_batch.to(device, non_blocking=True)
                if use_amp:
                    with autocast():
                        loss, loss_dict = criterion(model, noisy)
                else:
                    loss, loss_dict = criterion(model, noisy)
                val_losses.append(loss.item())
                for k in val_loss_details.keys():
                    if k in loss_dict:
                        val_loss_details[k].append(loss_dict[k])

        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_details = {k: (sum(v)/len(v) if v else 0.0)
                           for k, v in val_loss_details.items()}

        # TensorBoard epoch stats
        writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Console summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"Train loss : {avg_train_loss:.4f}")
        print(f"Val loss   : {avg_val_loss:.4f}")
        if avg_train_details['n2n_total'] != 0:
            print(f"  N2N      : {avg_train_details['n2n_total']:.4f}")
        if avg_train_details['wavelet_weighted'] != 0:
            print(f"  Wavelet  : {avg_train_details['wavelet_weighted']:.4f}")
        if avg_train_details['estimated_noise'] > 0:
            print(f"  Noise    : {avg_train_details['estimated_noise']*400:.1f} HU")
        if avg_train_details['adaptive_weight'] > 0:
            print(f"  Adapt W  : {avg_train_details['adaptive_weight']:.5f}")

        if avg_train_details['balance_ratio'] != 0:
            ratio = avg_train_details['balance_ratio']
            print(f"  Balance  : {ratio:.2f} (target ~{target_balance:.1f})")
            if config['monitoring'].get('warn_if_imbalanced', True):
                if ratio < target_balance * 0.5:
                    print("  -> Wavelet too strong")
                    balance_warnings += 1
                elif ratio > target_balance * 2.0:
                    print("  -> Wavelet too weak")
                    balance_warnings += 1
                else:
                    print("  -> Balance OK")
        print(f"LR         : {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}\n")

        # ---------------- Checkpoint ----------------
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss

        if epoch % config['training']['save_interval'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_val_loss,
                ckpt_dir / f"model_epoch_{epoch}.pth",
                is_best=is_best
            )
            cleanup_old_checkpoints(ckpt_dir, config['training']['keep_last_n'])

        # ---------------- Sample visualization ----------------
        if epoch % config['training']['sample_interval'] == 0 or epoch == 1:
            sample_seed = epoch // 10
            fixed_samples, slice_info = load_fixed_full_slices(
                config['data']['nc_ct_dir'],
                config['preprocessing']['hu_window'],
                seed=sample_seed
            )

            model.eval()
            with torch.no_grad():
                denoised_list = []
                metrics_list = []

                for i, single_sample in enumerate(fixed_samples):
                    single_noisy = single_sample.to(device)

                    # N2N input g1만 사용해서 denoise
                    g1, _ = criterion.n2n_loss.generate_subimages_checkerboard(single_noisy)
                    if use_amp:
                        with autocast():
                            single_denoised = model(g1)
                    else:
                        single_denoised = model(g1)
                    single_denoised = torch.clamp(single_denoised, 0, 1)
                    denoised_list.append(single_denoised)

                    # Adaptive metrics (내부에서 중앙 crop + sigma 추정)
                    if hasattr(criterion, 'compute_sample_metrics'):
                        m = criterion.compute_sample_metrics(
                            single_noisy,
                            slice_info=[slice_info[i]]
                        )[0]
                        metrics_list.append(m)

                        print(f"\n🔍 DEBUG Sample {i} ({m['label']}):")
                        print(f"   Input shape: {single_noisy.shape}")
                        print(f"   Input std  : {single_noisy.std():.4f}")
                        print(f"   est_noise  : {m['estimated_noise_hu']:.2f} HU")
                        print(f"   adapt_w    : {m['adaptive_weight']:.5f}")

                noisy_batch = torch.cat(fixed_samples, dim=0).to(device)
                denoised_batch = torch.cat(denoised_list, dim=0)

                if len(metrics_list) >= 2:
                    print(f"\n📊 Adaptive Metrics (Epoch {epoch}):")
                    hn, ln = metrics_list[0], metrics_list[1]
                    print(f"   HN: weight={hn['adaptive_weight']:.5f}, "
                          f"noise={hn['estimated_noise_hu']:.1f} HU, "
                          f"ratio={hn['noise_ratio']:.2f}x")
                    print(f"   LN: weight={ln['adaptive_weight']:.5f}, "
                          f"noise={ln['estimated_noise_hu']:.1f} HU, "
                          f"ratio={ln['noise_ratio']:.2f}x")
                    if hn['adaptive_weight'] > 0 and ln['adaptive_weight'] > 0:
                        wr = hn['adaptive_weight'] / ln['adaptive_weight']
                        print(f"   Weight Ratio (HN/LN): {wr:.2f}x")

                save_sample_images(
                    noisy_batch,
                    denoised_batch,
                    sample_dir / f"epoch_{epoch}.png",
                    epoch,
                    metrics=metrics_list if metrics_list else None
                )
            model.train()

        # Early stopping
        if early_stopping(avg_val_loss):
            print(f"\nEarly stopping at epoch {epoch}, best val loss {best_val_loss:.4f}")
            break

    writer.close()
    print("\n==================== Training Done ====================")
    print(f"Experiment dir: {exp_dir}")
    print(f"Best val loss : {best_val_loss:.4f}")
    if balance_warnings > 0:
        print(f"Balance warnings: {balance_warnings} (consider tuning wavelet_weight)")

if __name__ == '__main__':
    train_n2n()
