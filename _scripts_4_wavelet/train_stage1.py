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

# Add SwinIR to path
sys.path.insert(0, r'E:\LD-CT SR\_externals\SwinIR')
from models.network_swinir import SwinIR

from dataset import CTDenoiseDataset
from losses import CombinedLoss, SelfSupervisedCombinedLoss
from utils import (
    load_config, save_checkpoint, load_checkpoint, save_sample_images,
    cleanup_old_checkpoints, EarlyStopping, WarmupScheduler
)

def train_stage1():
    print("="*80)
    print("ðŸš€ Stage 1: Pretrain on External Low-Dose â†’ Full-Dose Dataset")
    print("="*80)
    
    # Load config
    script_dir = Path(__file__).parent
    config_path = script_dir / 'config.yaml'
    config = load_config(config_path)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nâœ… Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Output dirs
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config['data']['output_dir']) / f'stage1_pretrain_{timestamp}'
    ckpt_dir = exp_dir / 'checkpoints'
    log_dir = exp_dir / 'logs'
    sample_dir = exp_dir / 'samples'
    
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"\nðŸ“Š TensorBoard: tensorboard --logdir={log_dir}")
    
    # Training mode (define early for dataset)
    training_mode = config['training'].get('mode', 'supervised')
    print(f"\n📊 Training Mode: {training_mode.upper()}")
    
    # Dataset
    print("\nðŸ“Š Preparing dataset...")
    full_dataset = CTDenoiseDataset(
        low_dose_dir=config['data']['low_dose_dir'],
        full_dose_dir=config['data']['full_dose_dir'],
        hu_window=config['preprocessing']['hu_window'],
        patch_size=config['preprocessing']['patch_size'],
        config_aug=config['training']['augmentation'],
        mode='train',
        self_supervised=(training_mode == 'self_supervised')
    )
    
    val_size = int(len(full_dataset) * config['training']['val_split'])
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
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
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Train batches per epoch: {len(train_loader)}")
    
    # Model
    print("\nðŸ—ï¸  Building model...")
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
    
    # -------------------------------------------------------------------------
    # â­ FIXED â­ pretrained_path ìžë™ ë³´ì •
    # -------------------------------------------------------------------------
    pretrained_path = config['model']['pretrained']

    if pretrained_path is None or pretrained_path.strip() == "":
        pretrained_path = r"E:\LD-CT SR\Weights\001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
        print(f"   âš ï¸ Config pretrained path empty â†’ Using default:\n      {pretrained_path}")

    pretrained_path = Path(pretrained_path)

    print(f"   Loading pretrained:\n      {pretrained_path}")

    if pretrained_path.exists():
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
            print(f"   âœ… Loaded {len(pretrained_dict)} pretrained layers.")
        except Exception as e:
            print(f"   âš ï¸ Failed to load pretrained: {e}")
    else:
        print(f"   âš ï¸  Pretrained path NOT found. Skipping pretrained load.")
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Loss Function Selection based on mode
    # -------------------------------------------------------------------------
    learn_weights = config['training'].get('learn_weights', False)
    
    print(f"\n📊 Training Mode: {training_mode.upper()}")
    
    if training_mode == 'self_supervised':
        # Self-Supervised Loss (NC-CT용 - Ground Truth 불필요!)
        criterion = SelfSupervisedCombinedLoss(
            n2v_weight=config['training']['self_supervised_weights']['n2v'],
            wavelet_weight=config['training']['self_supervised_weights']['wavelet_sparsity'],
            tv_weight=config['training']['self_supervised_weights']['tv'],
            wavelet_threshold=config['training'].get('wavelet_threshold', 50),
            wavelet_levels=config['training'].get('wavelet_levels', 3)
        ).to(device)
        
        print("   ✅ Self-Supervised Loss 활성화")
        print("   → Ground Truth 불필요!")
        print("   → NC-CT 데이터에서 바로 학습 가능!")
        
    else:
        # Supervised Loss (기존 방식 - External data용)
        criterion = CombinedLoss(
            l1_weight=config['training']['loss_weights']['l1'],
            ssim_weight=config['training']['loss_weights']['ssim'],
            wavelet_weight=config['training']['loss_weights']['wavelet'],
            wavelet_threshold=config['training'].get('wavelet_threshold', 50),
            learn_weights=learn_weights
        ).to(device)
        
        print("   ✅ Supervised Loss 활성화")
        print("   → Ground Truth 필요")
        print("   → External paired data용")
    # -------------------------------------------------------------------------
    
    # Optimizer ---------------------------------------------------------------
    if training_mode == 'supervised' and learn_weights:
        params_to_optimize = [
            {'params': model.parameters(), 'lr': config['training']['learning_rate']},
            {'params': criterion.parameters(), 'lr': 0.01}
        ]
    else:
        params_to_optimize = model.parameters()

    if config['training']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config['training']['learning_rate'],
            betas=config['training']['betas'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=config['training']['learning_rate'],
            betas=config['training']['betas'],
            weight_decay=config['training']['weight_decay']
        )
    # -------------------------------------------------------------------------

    # Scheduler
    if config['training']['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training']['T_0'],
            T_mult=config['training']['T_mult'],
            eta_min=config['training']['eta_min']
        )
    else:
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
    from torch.amp import GradScaler as AmpGradScaler
    scaler = AmpGradScaler('cuda') if use_amp else None
    
    # Resume
    start_epoch = 1
    best_val_loss = float('inf')
    if config['training']['resume']:
        resume_path = Path(config['training']['resume'])
        if resume_path.exists():
            start_epoch, _ = load_checkpoint(resume_path, model, optimizer, scheduler)
            start_epoch += 1
            print(f"âœ… Resumed from epoch {start_epoch-1}")

    # Training Loop ============================================================
    print("\n" + "="*80)
    print("ðŸš€ Starting training...")
    print("="*80 + "\n")
    
    global_step = 0
    
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        model.train()
        train_losses = []
        if training_mode == 'self_supervised':
            train_loss_details = {'n2v': [], 'wavelet_sparsity': [], 'tv': []}
        else:
            train_loss_details = {'l1': [], 'ssim': [], 'wavelet': []}
        
        
        if warmup.is_warmup():
            warmup.step()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['num_epochs']}")
        
        for batch_idx, (input_img, target_img) in enumerate(pbar):
            input_img = input_img.to(device, non_blocking=True)
            target_img = target_img.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    pred = model(input_img)
                    pred = torch.clamp(pred, 0, 1)
                    # Loss 계산 (self-supervised: input==target, supervised: input!=target)
                    loss, loss_dict = criterion(pred, target_img)
                
                scaler.scale(loss).backward()
                
                if config['training']['gradient_clip'] > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config['training']['gradient_clip'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(input_img)
                pred = torch.clamp(pred, 0, 1)
                # Loss 계산 (self-supervised: input==target, supervised: input!=target)
                loss, loss_dict = criterion(pred, target_img)
                
                loss.backward()
                
                if config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config['training']['gradient_clip'])
                
                optimizer.step()
            
            train_losses.append(loss.item())
            for key in train_loss_details:
                train_loss_details[key].append(loss_dict[key])
            
            # TensorBoard
            global_step += 1
            writer.add_scalar('Train/loss_total', loss.item(), global_step)
            
            # Mode-specific loss logging
            if training_mode == 'self_supervised':
                writer.add_scalar('Train/loss_n2v', loss_dict['n2v'], global_step)
                writer.add_scalar('Train/loss_wavelet_sparsity', loss_dict['wavelet_sparsity'], global_step)
                writer.add_scalar('Train/loss_tv', loss_dict['tv'], global_step)
            else:
                writer.add_scalar('Train/loss_l1', loss_dict['l1'], global_step)
                writer.add_scalar('Train/loss_ssim', loss_dict['ssim'], global_step)
                writer.add_scalar('Train/loss_wavelet', loss_dict['wavelet'], global_step)
                
                # Learnable weight logging
                if learn_weights:
                    writer.add_scalar('Train/weight_ssim', loss_dict['weight_ssim'], global_step)
                    writer.add_scalar('Train/weight_wavelet', loss_dict['weight_wavelet'], global_step)

            # Progress bar postfix
            if training_mode == 'self_supervised':
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'n2v': f"{loss_dict['n2v']:.4f}",
                    'wav_sp': f"{loss_dict['wavelet_sparsity']:.4f}",
                    'tv': f"{loss_dict['tv']:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
            else:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'l1': f"{loss_dict['l1']:.4f}",
                    'ssim': f"{loss_dict['ssim']:.4f}",
                    'wav': f"{loss_dict['wavelet']:.4f}",
                    'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
                })
        
        if not warmup.is_warmup():
            scheduler.step()
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_details = {k: sum(v)/len(v) for k, v in train_loss_details.items()}
        
        # Validation -----------------------------------------------------------
        model.eval()
        val_losses = []
        if training_mode == 'self_supervised':
            val_loss_details = {'n2v': [], 'wavelet_sparsity': [], 'tv': []}
        else:
            val_loss_details = {'l1': [], 'ssim': [], 'wavelet': []}
        
        with torch.no_grad():
            for input_img, target_img in tqdm(val_loader, desc="Validation", leave=False):
                input_img = input_img.to(device, non_blocking=True)
                target_img = target_img.to(device, non_blocking=True)
                
                with autocast() if use_amp else torch.no_grad():
                    pred = model(input_img)
                    pred = torch.clamp(pred, 0, 1)
                    # Loss 계산
                    loss, loss_dict = criterion(pred, target_img)
                
                val_losses.append(loss.item())
                for key in val_loss_details:
                    val_loss_details[key].append(loss_dict[key])
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_details = {k: sum(v)/len(v) for k, v in val_loss_details.items()}
        
        writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
        writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
        
        print(f"\n{'='*80}")
        print(f"ðŸ“Š Epoch {epoch} Summary:")
        print(f"{'='*80}")
        if training_mode == 'self_supervised':
            print(f"Train Loss: {avg_train_loss:.4f} | N2V: {avg_train_details['n2v']:.4f} | Wav: {avg_train_details['wavelet_sparsity']:.4f} | TV: {avg_train_details['tv']:.4f}")
            print(f"Val Loss:   {avg_val_loss:.4f} | N2V: {avg_val_details['n2v']:.4f} | Wav: {avg_val_details['wavelet_sparsity']:.4f} | TV: {avg_val_details['tv']:.4f}")
        else:
            print(f"Train Loss: {avg_train_loss:.4f} | L1: {avg_train_details['l1']:.4f} | SSIM: {avg_train_details['ssim']:.4f} | Wav: {avg_train_details['wavelet']:.4f}")
            print(f"Val Loss:   {avg_val_loss:.4f} | L1: {avg_val_details['l1']:.4f} | SSIM: {avg_val_details['ssim']:.4f} | Wav: {avg_val_details['wavelet']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*80}\n")
        
        # Save checkpoint
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
        
        if epoch % config['training']['sample_interval'] == 0:
            model.eval()
            with torch.no_grad():
                input_sample, target_sample = next(iter(val_loader))
                input_sample = input_sample.to(device)
                target_sample = target_sample.to(device)
                
                with autocast() if use_amp else torch.no_grad():
                    pred_sample = model(input_sample)
                    pred_sample = torch.clamp(pred_sample, 0, 1)
                
                save_sample_images(
                    input_sample, pred_sample, target_sample,
                    sample_dir / f"epoch_{epoch}.png",
                    epoch
                )
        
        if early_stopping(avg_val_loss):
            print(f"\nðŸ›‘ Early stopping triggered at epoch {epoch}")
            print(f"   Best val loss: {best_val_loss:.4f}")
            break
    
    writer.close()
    
    print("\n" + "="*80)
    print("âœ… Training completed!")
    print("="*80)
    print(f"ðŸ“ Experiment directory: {exp_dir}")
    print(f"ðŸ“ Checkpoints: {ckpt_dir}")
    print(f"ðŸ“ Samples: {sample_dir}")
    print(f"ðŸ“Š Best val loss: {best_val_loss:.4f}")
    print(f"ðŸ“Š TensorBoard: tensorboard --logdir={log_dir}")


if __name__ == '__main__':
    train_stage1()