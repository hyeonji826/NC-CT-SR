# train_n2n.py - Self-Supervised Training with Neighbor2Neighbor + Wavelet

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import sys
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import yaml
import argparse

# Add SwinIR to path
sys.path.insert(0, r'E:\LD-CT SR\_externals\SwinIR')
from models.network_swinir import SwinIR

from dataset_n2n import NCCTDenoiseDataset
from losses_n2n import CombinedN2NWaveletLoss, Neighbor2NeighborLoss
from utils import (
    load_config, save_checkpoint, load_checkpoint, save_sample_images,
    cleanup_old_checkpoints, EarlyStopping, WarmupScheduler
)

def load_fixed_full_slices(nc_ct_dir, hu_window):
    """
    HIGH-NOISE(HN) + RELATIVELY-LOW-NOISE(LN) full slices 선택
    
    ⚠️ 전체 데이터셋이 noisy하므로 RELATIVE 기준 사용
    - HN: 90th percentile 이상 (극단적 high)
    - LN: 10th percentile 이하 (상대적 low) + high edge
    """
    import nibabel as nib
    import numpy as np
    from pathlib import Path
    from scipy import ndimage

    nc_ct_dir = Path(nc_ct_dir)
    files = sorted(list(nc_ct_dir.glob("*.nii.gz")))

    candidates = []

    print("\n🔍 Analyzing slices (RELATIVE noise-based selection)...")

    # ✅ 전체 파일 스캔 (충분한 샘플 확보)
    for file_idx, file_path in enumerate(files):
        try:
            nii = nib.load(str(file_path))
            volume = nii.get_fdata()

            D = volume.shape[2]
            # 많은 슬라이스 샘플링
            slice_indices = [D//5, D//3, D//2, 2*D//3, 4*D//5]

            for slice_idx in slice_indices:
                if slice_idx >= D:
                    continue

                slice_2d = volume[:, :, slice_idx]

                # 중심부(복부)만 사용
                h, w = slice_2d.shape
                center_ratio_w = 0.55
                center_ratio_h = 0.60
                margin_h = int(h * (1 - center_ratio_h) / 2)
                margin_w = int(w * (1 - center_ratio_w) / 2)

                center_slice = slice_2d[margin_h:h - margin_h,
                                        margin_w:w - margin_w]

                # 조직(HU -100 ~ 100)만 보고 noise 계산
                tissue_mask = (center_slice > -100) & (center_slice < 100)
                if tissue_mask.sum() < 1000:
                    continue

                tissue_region = center_slice[tissue_mask]
                noise_std = tissue_region.std()

                # Sobel edge
                gx = ndimage.sobel(center_slice, axis=0)
                gy = ndimage.sobel(center_slice, axis=1)
                edge_mag = np.hypot(gx, gy)
                edge_score = edge_mag[tissue_mask].mean()

                candidates.append({
                    'file_path': file_path,
                    'slice_idx': slice_idx,
                    'noise_std': float(noise_std),
                    'edge_score': float(edge_score),
                    'slice_2d': center_slice
                })
        except Exception as e:
            print(f"  ⚠️ Skip {file_path.name}: {e}")
            continue
        
        # Progress
        if (file_idx + 1) % 50 == 0:
            print(f"  Scanned {file_idx + 1}/{len(files)} files...")

    if len(candidates) < 2:
        raise RuntimeError("Not enough valid slices found!")

    # ✅ Percentile 기준 계산
    all_noise = np.array([c['noise_std'] for c in candidates])
    p10 = np.percentile(all_noise, 10)
    p90 = np.percentile(all_noise, 90)
    median = np.median(all_noise)
    
    print(f"   Noise distribution: 10th={p10:.1f}, median={median:.1f}, 90th={p90:.1f} HU")

    # 1) HN: 90th percentile 이상 중 가장 높은 것
    hn_candidates = [c for c in candidates if c['noise_std'] >= p90]
    high_noise = max(hn_candidates, key=lambda x: x['noise_std'])

    # 2) LN: 10th percentile 이하 중 edge 가장 높은 것
    ln_candidates = [c for c in candidates if c['noise_std'] <= p10]
    
    # Fallback: 10th percentile 이하가 없으면 하위 15%
    if len(ln_candidates) < 3:
        candidates_sorted = sorted(candidates, key=lambda x: x['noise_std'])
        ln_candidates = candidates_sorted[:len(candidates)//7]
    
    low_noise = max(ln_candidates, key=lambda x: x.get('edge_score', 0.0))

    print("✅ Selected 2 representative slices (RELATIVE extremes):")
    print(f"   [HN] {high_noise['file_path'].name[:30]} "
          f"slice {high_noise['slice_idx']}  "
          f"Noise: {high_noise['noise_std']:.1f} HU (top {100*(1 - high_noise['noise_std']/all_noise.max()):.1f}%)  "
          f"Edge: {high_noise.get('edge_score', 0):.1f}")
    print(f"   [LN] {low_noise['file_path'].name[:30]} "
          f"slice {low_noise['slice_idx']}  "
          f"Noise: {low_noise['noise_std']:.1f} HU (bottom {100*low_noise['noise_std']/all_noise.max():.1f}%)  "
          f"Edge: {low_noise.get('edge_score', 0):.1f}")
    print(f"   Noise ratio (HN/LN): {high_noise['noise_std'] / max(low_noise['noise_std'], 1e-6):.2f}x")
    print(f"   ⚠️  Note: Entire dataset is noisy (10th %ile = {p10:.1f} HU)")

    # 텐서 변환
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='_scripts_4_wavelet/config/config_n2n.yaml')
    parser.add_argument('--exp', type=str, default='debug')
    args = parser.parse_args()
    return args

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def train_n2n():
    print("="*80)
    print(" Self-Supervised Training: Neighbor2Neighbor + Wavelet Sparsity")
    print("="*80)
    print("\n Key Features:")
    print("   [OK] NO paired data required (self-supervised!)")
    print("   [OK] N2N: Main denoising mechanism")
    print("   [OK] Wavelet: Regularization (prevents overfitting)")
    print("   [OK] Balance: N2N >> Wavelet (20:1)")
    print("="*80)
    
    # ---- 1) 인자 파싱 ----
    args = parse_args()

    # ---- 2) config 경로 결정 ----
    script_dir = Path(__file__).parent

    # args.config가 relative path면, 스크립트 기준으로 붙여주기
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    print(f"\n📄 Using config file: {config_path}")

    # ---- 3) config 로드 ----
    config = load_config(config_path)
    exp_name = args.exp
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Output dirs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config['data']['output_dir']) / f'n2n_wavelet_{timestamp}'
    ckpt_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    sample_dir = exp_dir / 'samples'
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    writer = SummaryWriter(log_dir)
    print(f"\n TensorBoard: tensorboard --logdir={log_dir}")
    
    # Dataset (NC-CT only!)
    print("\nÃ°Å¸â€œâ€š Loading NC-CT dataset (self-supervised)...")
    full_dataset = NCCTDenoiseDataset(
        nc_ct_dir=config['data']['nc_ct_dir'],
        hu_window=config['preprocessing']['hu_window'],
        patch_size=config['preprocessing']['patch_size'],
        config_aug=config['training']['augmentation'],
        mode='train'
    )
    
    # 8:1:1 split (train:val:test)
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
        persistent_workers=True if config['training']['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Train samples: {len(train_dataset)} (80%)")
    print(f"   Val samples: {len(val_dataset)} (10%)")
    print(f"   Test samples: {len(test_dataset)} (10%)")
    print(f"   Batches per epoch: {len(train_loader)}")
    
    # Save test set indices for later inference
    test_indices_path = exp_dir / 'test_indices.txt'
    with open(test_indices_path, 'w') as f:
        for idx in test_dataset.indices:
            file_idx = idx % len(full_dataset.files)
            f.write(f"{full_dataset.files[file_idx]}\n")
    print(f"   Test set file list saved: {test_indices_path}")
    
    # Prepare fixed validation samples for consistent progress tracking
    print("\nPreparing fixed validation samples (HN + LN comparison)...")
    fixed_samples, slice_info = load_fixed_full_slices(
        config['data']['nc_ct_dir'],
        config['preprocessing']['hu_window']
    )
    print(f"   Fixed samples: {len(fixed_samples)} full slices (HN + LN)")
    print(f"   Size: {fixed_samples[0].shape}")
    
    # Model
    print("\n  Building SwinIR model...")
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
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Pretrained weights
    pretrained_path = config['model'].get('pretrained')
    if pretrained_path and Path(pretrained_path).exists():
        print(f"\n Loading pretrained weights:")
        print(f"   {pretrained_path}")
        try:
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if 'params' in pretrained_dict:
                pretrained_dict = pretrained_dict['params']
            elif 'model_state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict['model_state_dict']
            
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                             if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"    Loaded {len(pretrained_dict)}/{len(model_dict)} layers")
        except Exception as e:
            print(f"     Failed to load pretrained: {e}")
    
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
            # ✅ CRITICAL: Pass YAML parameters to criterion (S1A/S1B 차이 반영)
            target_noise=config['training'].get('target_noise', 0.15),
            adaptive_weight_range=tuple(config['training'].get('adaptive_weight_range', [0.3, 3.0])),
            edge_weight=config['training'].get('edge_weight', 0.05),
        ).to(device)
        
        print(f"\n✅ Criterion Parameters from YAML:")
        print(f"   target_noise         : {config['training'].get('target_noise', 0.15)}")
        print(f"   adaptive_weight_range: {config['training'].get('adaptive_weight_range', [0.3, 3.0])}")
        print(f"   edge_weight          : {config['training'].get('edge_weight', 0.05)}")
        
    else:  # n2n_only (fallback)
        criterion = Neighbor2NeighborLoss(
            gamma=config['training']['n2n_gamma']
        ).to(device)
        print(f"\n⚠️  Using basic N2N loss (no wavelet/edge)")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
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
    
    # Mixed Precision
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
            print(f"\n Resumed from epoch {start_epoch-1}")
            print(f"   Best val loss: {best_val_loss:.4f}")
        else:
            print(f"\n WARNING: Resume path not found: {resume_path}")
            print(f"   Starting from scratch...")
    
    # Training Loop
    print("\n" + "="*80)
    print(" Starting Self-Supervised Training...")
    print("="*80 + "\n")
    
    global_step = 0
    
    # Balance monitoring
    balance_warnings = 0
    target_balance = config['monitoring'].get('target_balance_ratio', 20.0)
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        model.train()
        train_losses = []
        train_loss_details = {
            'n2n_rec': [], 'n2n_reg': [], 'n2n_total': [],
            'wavelet': [], 'total': [], 'balance_ratio': [],
            'estimated_noise': [], 'adaptive_weight': []
        }
        
        if warmup.is_warmup():
            warmup.step()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['num_epochs']}")
        
        for batch_idx, noisy_batch in enumerate(pbar):
            # N2N: Only need noisy input! (no target)
            noisy = noisy_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    # N2N loss handles model forward internally
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
                # N2N loss handles model forward internally
                loss, loss_dict = criterion(model, noisy)
                
                loss.backward()
                
                if config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config['training']['gradient_clip']
                    )
                
                optimizer.step()
            
            # Record losses
            train_losses.append(loss.item())
            for key in train_loss_details:
                if key in loss_dict:
                    train_loss_details[key].append(loss_dict[key])
            
            # TensorBoard logging
            global_step += 1
            writer.add_scalar('Train/total_loss', loss.item(), global_step)
            
            if 'n2n_rec' in loss_dict:
                writer.add_scalar('Train/n2n_rec', loss_dict['n2n_rec'], global_step)
                writer.add_scalar('Train/n2n_reg', loss_dict['n2n_reg'], global_step)
                writer.add_scalar('Train/n2n_total', loss_dict['n2n_total'], global_step)
            
            if 'wavelet_weighted' in loss_dict:
                writer.add_scalar('Train/wavelet', loss_dict['wavelet_weighted'], global_step)
            
            if 'balance_ratio' in loss_dict:
                writer.add_scalar('Train/balance_ratio', loss_dict['balance_ratio'], global_step)
            
            # Adaptive metrics (if available)
            if 'estimated_noise' in loss_dict:
                writer.add_scalar('Train/estimated_noise', loss_dict['estimated_noise'], global_step)
            
            if 'adaptive_weight' in loss_dict:
                writer.add_scalar('train/adaptive_weight', loss_dict.get('adaptive_weight', 0), global_step)
                writer.add_scalar('train/edge_loss', loss_dict.get('edge_loss', 0), global_step)
            
            # Update progress bar
            pbar_dict = {
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            }
            if 'n2n_total' in loss_dict:
                pbar_dict['n2n'] = f"{loss_dict['n2n_total']:.4f}"
            if 'wavelet_weighted' in loss_dict:
                pbar_dict['wav'] = f"{loss_dict['wavelet_weighted']:.4f}"
            if 'balance_ratio' in loss_dict:
                pbar_dict['bal'] = f"{loss_dict['balance_ratio']:.1f}"
            if 'estimated_noise' in loss_dict and loss_dict['estimated_noise'] > 0:
                pbar_dict['noise'] = f"{loss_dict['estimated_noise']*400:.1f}"  # HU scale
            
            pbar.set_postfix(pbar_dict)
        
        if not warmup.is_warmup():
            scheduler.step()
        
        # Epoch statistics
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_details = {
            k: sum(v)/len(v) for k, v in train_loss_details.items() if len(v) > 0
        }
        
        # Validation
        model.eval()
        val_losses = []
        val_loss_details = {
            'n2n_rec': [], 'n2n_reg': [], 'n2n_total': [],
            'wavelet': [], 'total': [], 'balance_ratio': [],
            'estimated_noise': [], 'adaptive_weight': []
        }
        
        with torch.no_grad():
            for noisy_batch in tqdm(val_loader, desc="Validation", leave=False):
                noisy = noisy_batch.to(device, non_blocking=True)
                
                if use_amp:
                    with autocast():
                        loss, loss_dict = criterion(model, noisy)
                else:
                    loss, loss_dict = criterion(model, noisy)
                
                val_losses.append(loss.item())
                for key in val_loss_details:
                    if key in loss_dict:
                        val_loss_details[key].append(loss_dict[key])
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_details = {
            k: sum(v)/len(v) for k, v in val_loss_details.items() if len(v) > 0
        }

        
        # Log to TensorBoard
        writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
        writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f" Epoch {epoch} Summary:")
        print(f"{'='*80}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        if 'n2n_total' in avg_train_details:
            print(f"  |- N2N:     {avg_train_details['n2n_total']:.4f}")
            print(f"  |   |- Rec: {avg_train_details['n2n_rec']:.4f}")
            print(f"  |   \\- Reg: {avg_train_details['n2n_reg']:.4f}")
        if 'wavelet' in avg_train_details:
            print(f"  \\- Wavelet: {avg_train_details['wavelet']:.4f}")
        
        print(f"\nVal Loss:   {avg_val_loss:.4f}")
        if 'n2n_total' in avg_val_details:
            print(f"  |- N2N:     {avg_val_details['n2n_total']:.4f}")
        if 'wavelet' in avg_val_details:
            print(f"  \\- Wavelet: {avg_val_details['wavelet']:.4f}")
        # Adaptive metrics
        if 'estimated_noise' in avg_train_details and avg_train_details['estimated_noise'] > 0:
            noise_hu = avg_train_details['estimated_noise'] * 400
            print(f"\nEstimated Noise: {noise_hu:.1f} HU")
        
        if 'adaptive_weight' in avg_train_details:
            print(f"Adaptive Weight: {avg_train_details['adaptive_weight']:.6f}")
        
        # Balance check
        if 'balance_ratio' in avg_train_details:
            ratio = avg_train_details['balance_ratio']
            print(f"\nBalance Ratio: {ratio:.2f} (target: ~{target_balance:.1f})")
            
            if config['monitoring'].get('warn_if_imbalanced', True):
                if ratio < target_balance * 0.5:
                    print(f"  WARNING: Wavelet too strong! (ratio < {target_balance*0.5:.1f})")
                    balance_warnings += 1
                elif ratio > target_balance * 2.0:
                    print(f"  WARNING: Wavelet too weak! (ratio > {target_balance*2.0:.1f})")
                    balance_warnings += 1
                else:
                    print(f"  Balance is good!")
        
        print(f"\nLR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}\n")
        
        # Check if best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, avg_val_loss,
                ckpt_dir / f"model_epoch_{epoch}.pth",
                is_best=is_best
            )
            cleanup_old_checkpoints(ckpt_dir, config['training']['keep_last_n'])
        
        # Save samples (use FIXED samples for consistent tracking)
        if epoch % config['training']['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                # Process all fixed samples as batch
                noisy_batch = torch.cat(fixed_samples, dim=0).to(device)
                
                # N2N forward (use g1 for inference)
                g1, g2 = criterion.n2n_loss.generate_subimages_checkerboard(noisy_batch)
                
                if use_amp:
                    with autocast():
                        denoised_batch = model(g1)
                else:
                    denoised_batch = model(g1)
                
                denoised_batch = torch.clamp(denoised_batch, 0, 1)
                
                # Compute per-sample adaptive metrics
                sample_metrics = None
                if hasattr(criterion, "compute_sample_metrics"):
                    sample_metrics = criterion.compute_sample_metrics(
                        noisy_batch, 
                        slice_info=slice_info
                    )
                    
                    # Debug output
                    if len(sample_metrics) >= 2:
                        print(f"\n📊 Adaptive Metrics (Epoch {epoch}):")
                        print(f"   HN: weight={sample_metrics[0]['adaptive_weight']:.5f}, "
                              f"noise={sample_metrics[0]['estimated_noise_hu']:.1f} HU, "
                              f"ratio={sample_metrics[0]['noise_ratio']:.2f}x")
                        print(f"   LN: weight={sample_metrics[1]['adaptive_weight']:.5f}, "
                              f"noise={sample_metrics[1]['estimated_noise_hu']:.1f} HU, "
                              f"ratio={sample_metrics[1]['noise_ratio']:.2f}x")
                        
                        if sample_metrics[0]['adaptive_weight'] > 0 and sample_metrics[1]['adaptive_weight'] > 0:
                            ratio = sample_metrics[0]['adaptive_weight'] / sample_metrics[1]['adaptive_weight']
                            print(f"   Weight Ratio (HN/LN): {ratio:.2f}x")

                # Save comparison with metrics
                save_sample_images(
                    noisy_batch,
                    denoised_batch,
                    sample_dir / f"epoch_{epoch}.png",
                    epoch,
                    metrics=sample_metrics
                )
            
            model.train()
        
        # Early stopping
        if early_stopping(avg_val_loss):
            print(f"\n Early stopping triggered at epoch {epoch}")
            print(f"   Best val loss: {best_val_loss:.4f}")
            break
    
    writer.close()
    
    # Final summary
    print("\n" + "="*80)
    print(" Training Completed!")
    print("="*80)
    print(f" Experiment directory: {exp_dir}")
    print(f" Checkpoints: {ckpt_dir}")
    print(f" Samples: {sample_dir}")
    print(f" Best val loss: {best_val_loss:.4f}")
    print(f" TensorBoard: tensorboard --logdir={log_dir}")
    
    if balance_warnings > 0:
        print(f"\n  Balance warnings: {balance_warnings}")
        print(f"   Consider adjusting wavelet_weight in config")
    
    print("="*80)


if __name__ == '__main__':
    train_n2n()