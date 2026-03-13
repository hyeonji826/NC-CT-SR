"""
train.py - MA-HybridNet 학습 파이프라인

모드: ma_hybrid (Medical-Aware Hybrid Denoising Network)

실행:
  python train.py --config config_ma_hybrid.yaml --mode ma_hybrid

"""

import os
import sys
import argparse
import math
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils import (
    setup_logger, save_checkpoint,
    compute_psnr, compute_ssim,
    save_sample_images,
)


# ============================================================
# YAML 설정 로드
# ============================================================

def load_config(config_path: str) -> dict:
    """YAML 설정 파일을 로드하여 dict로 반환."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


# ============================================================
# 디렉토리 자동 생성
# ============================================================

def setup_directories(output_root: str) -> dict:
    """실행 시 출력 폴더를 자동 생성."""
    dirs = {
        'root': output_root,
        'weights': os.path.join(output_root, 'weights'),
        'samples': os.path.join(output_root, 'samples'),
        'logs': os.path.join(output_root, 'logs'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ============================================================
# Medical-Aware Hybrid Network Trainer
# ============================================================

class MAHybridNetTrainer:
    """
    MA-HybridNet Trainer (Medical-Aware Hybrid Denoising Network).

    특징:
        - HU + Spatial + Frequency 다중 모달 특징 융합
        - Swin + NAFNet Hybrid Backbone
        - 의료 영상 특화 손실 함수 (NPS, Texture, Edge 등)
        - Noisier2Noise 전략
        - CE-CT gradient-domain texture discriminator (unpaired)
        - Inpaint 영역 신뢰도 감소 (mask-aware weighted loss)
    """

    def __init__(self, config: dict):
        from model import MAHybridNet
        from losses import MedicalAwareLoss, GradientTextureDiscriminator, compute_sobel_2ch

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compute_sobel_2ch = compute_sobel_2ch

        # Paths
        self.nifti_root = config['paths']['nifti_root']
        self.split_dir = config['paths']['split_dir']
        self.output_root = Path(config['paths']['output_root']) / 'ma_hybrid'
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Directories
        self.log_dir = self.output_root / 'logs'
        self.weights_dir = self.output_root / 'weights'
        self.samples_dir = self.output_root / 'samples'
        self.log_dir.mkdir(exist_ok=True)
        self.weights_dir.mkdir(exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)

        # Logger
        self.logger = setup_logger(str(self.log_dir), 'MAHybridNet')

        # Training config
        ma_cfg = config.get('ma_hybrid', {})
        self.epochs = ma_cfg.get('epochs', 200)
        self.batch_size = ma_cfg.get('batch_size', 2)
        self.crop_size = ma_cfg.get('crop_size', 512)
        self.lr = ma_cfg.get('lr', 2e-4)
        self.min_lr = ma_cfg.get('min_lr', 1e-6)
        self.weight_decay = ma_cfg.get('weight_decay', 1e-5)
        self.grad_clip_norm = ma_cfg.get('grad_clip_norm', 1.0)
        self.save_every = ma_cfg.get('save_every', 5)
        self.patience = ma_cfg.get('patience', 30)
        self.warmup_epochs = ma_cfg.get('warmup_epochs', 5)

        # Model config
        model_cfg = config.get('model', {})
        self.swin_embed_dim = model_cfg.get('embed_dim', 96)
        self.swin_depths = tuple(model_cfg.get('swin_depths', [2, 2]))
        self.swin_num_heads = tuple(model_cfg.get('num_heads', [3, 6]))
        self.nafnet_depth = model_cfg.get('nafnet_depth', 18)
        self.nafnet_width = model_cfg.get('nafnet_width', 384)

        # Loss config (Residual Learning)
        lambda_l1 = ma_cfg.get('lambda_l1', 1.0)
        lambda_noise_residual = ma_cfg.get('lambda_noise_residual', 2.0)
        lambda_ssim = ma_cfg.get('lambda_ssim', 2.0)
        lambda_edge = ma_cfg.get('lambda_edge', 1.5)
        lambda_lowfreq = ma_cfg.get('lambda_lowfreq', 2.0)
        lambda_nps = ma_cfg.get('lambda_nps', 0.5)
        lambda_texture = ma_cfg.get('lambda_texture', 1.0)
        lambda_artifact = ma_cfg.get('lambda_artifact', 1.0)
        lambda_uniformity = ma_cfg.get('lambda_uniformity', 0.5)

        # HU range
        data_cfg = config.get('data', {})
        self.hu_min = data_cfg.get('hu_min', -1000.0)
        self.hu_max = data_cfg.get('hu_max', 1000.0)

        # Noise model
        noise_cfg = config.get('noise', {})
        nps_root = noise_cfg.get('nps_root')
        target_std = ma_cfg.get('target_noise_std_hu', 30.0)
        beam_hardening = noise_cfg.get('beam_hardening_strength', 0.0)

        from nuclear_noise import NuclearNoiseModel
        self.noise_model = NuclearNoiseModel(
            nps_root=nps_root,
            target_noise_std_hu=target_std,
            hu_min=self.hu_min,
            hu_max=self.hu_max,
            beam_hardening_strength=beam_hardening,
            clip_dc=True
        )

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Noise model: std={target_std} HU")

        # Dataloaders
        processed_root = config.get('paths', {}).get('processed_root', '')
        self.processed_root = processed_root  # streak-corrected NIfTI root (fixed sample용)
        mask_root = config.get('paths', {}).get('mask_root', '')
        if processed_root and os.path.isdir(processed_root):
            # Action-Guided: raw input → streak-corrected target + mask guidance
            self.logger.info("Loading Action-Guided datasets (streak-corrected targets)...")
            if mask_root and os.path.isdir(mask_root):
                self.logger.info(f"  Action masks: {mask_root}")
            from dataset import get_action_guided_dataloaders
            self.train_loader, self.val_loader, self.test_loader = \
                get_action_guided_dataloaders(
                    raw_root=self.nifti_root,
                    processed_root=processed_root,
                    noise_model=self.noise_model,
                    split_dir=self.split_dir,
                    mask_root=mask_root,
                    crop_size=self.crop_size,
                    batch_size=self.batch_size,
                    num_workers=data_cfg.get('num_workers', 0),
                    seed=data_cfg.get('seed', 42)
                )
        else:
            # Fallback: original Noisier2Noise
            self.logger.info("Loading MA-Hybrid datasets (original N2N)...")
            from dataset import get_noisier_finetune_dataloaders
            self.train_loader, self.val_loader, self.test_loader = \
                get_noisier_finetune_dataloaders(
                    noisy_root=self.nifti_root,
                    noise_model=self.noise_model,
                    split_dir=self.split_dir,
                    crop_size=self.crop_size,
                    batch_size=self.batch_size,
                    num_workers=data_cfg.get('num_workers', 0),
                    seed=data_cfg.get('seed', 42)
                )

        self.logger.info(f"[train] Batches: {len(self.train_loader)}")
        self.logger.info(f"[val]   Batches: {len(self.val_loader)}")

        # Model
        self.logger.info("Building MA-HybridNet model...")
        self.model = MAHybridNet(
            img_size=self.crop_size,
            swin_embed_dim=self.swin_embed_dim,
            swin_depths=self.swin_depths,
            swin_num_heads=self.swin_num_heads,
            nafnet_depth=self.nafnet_depth,
            nafnet_width=self.nafnet_width,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Params: {total_params:,} | Trainable: {trainable_params:,}")

        # Texture Discriminator config
        td_cfg = config.get('texture_discriminator', {})
        self.use_discriminator = td_cfg.get('enabled', False)
        self.lambda_adversarial = td_cfg.get('lambda_adversarial', 0.01)
        inpaint_trust = td_cfg.get('inpaint_trust', 0.3)

        # Loss (Residual Learning + inpaint trust via |raw-processed| diff)
        self.criterion = MedicalAwareLoss(
            lambda_l1=lambda_l1,
            lambda_ssim=lambda_ssim,
            lambda_edge=lambda_edge,
            lambda_lowfreq=lambda_lowfreq,
            lambda_nps=lambda_nps,
            lambda_texture=lambda_texture,
            lambda_artifact=lambda_artifact,
            lambda_noise_residual=lambda_noise_residual,
            lambda_uniformity=lambda_uniformity,
            inpaint_trust=inpaint_trust,
        )

        # Optimizer (Generator)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # Texture Discriminator + CE-CT pool
        self.discriminator = None
        self.d_optimizer = None
        self.ce_pool = None

        self.d_grad_clip_norm = td_cfg.get('d_grad_clip_norm', 1.0)

        if self.use_discriminator:
            ce_dicom_root = config.get('paths', {}).get('ce_ct_dicom_root', '')
            if ce_dicom_root and os.path.isdir(ce_dicom_root):
                self.logger.info("Initializing Gradient-Domain Texture Discriminator...")

                # Discriminator
                ndf = td_cfg.get('ndf', 64)
                self.discriminator = GradientTextureDiscriminator(
                    in_channels=2, ndf=ndf
                ).to(self.device)
                d_params = sum(p.numel() for p in self.discriminator.parameters())
                self.logger.info(f"  Discriminator params: {d_params:,} (ndf={ndf})")

                # D optimizer (lower LR than G)
                d_lr = td_cfg.get('d_lr', 0.0001)
                d_weight_decay = td_cfg.get('d_weight_decay', 1e-5)
                d_betas = tuple(td_cfg.get('d_betas', [0.5, 0.999]))
                self.d_optimizer = torch.optim.AdamW(
                    self.discriminator.parameters(),
                    lr=d_lr,
                    weight_decay=d_weight_decay,
                    betas=d_betas,
                )

                # CE-CT texture patch pool
                pool_size = td_cfg.get('pool_size', 500)
                from dataset import CECTTexturePatchPool
                self.ce_pool = CECTTexturePatchPool(
                    dicom_root=ce_dicom_root,
                    pool_size=pool_size,
                    crop_size=self.crop_size,
                    hu_min=self.hu_min,
                    hu_max=self.hu_max,
                    seed=data_cfg.get('seed', 42),
                )

                if self.ce_pool.pool is None:
                    self.logger.warning("CE-CT pool failed, disabling discriminator")
                    self.use_discriminator = False
                    self.discriminator = None
                else:
                    self.logger.info(f"  lambda_adversarial: {self.lambda_adversarial}")
                    self.logger.info(f"  inpaint_trust: {inpaint_trust}")
            else:
                self.logger.warning(f"CE-CT DICOM root not found: {ce_dicom_root}")
                self.use_discriminator = False

        # Mixed precision scaler (비활성화 - LayerNorm이 FP16에서 불안정)
        self.scaler = None

        # LR scheduler (Warmup + Cosine Annealing)
        self.scheduler = self._create_scheduler()

        # Fixed sample for visualization
        self._load_fixed_sample()

        # Tracking
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # Resume from checkpoint
        resume_ckpt = ma_cfg.get('resume_ckpt', '')
        if resume_ckpt and os.path.exists(resume_ckpt):
            self._load_checkpoint(resume_ckpt)

        # WandB 초기화
        self.use_wandb = ma_cfg.get('use_wandb', True)
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="LD-CT-SR",
                    name=f"ma_hybrid_ep{self.start_epoch}",
                    config={
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'lr': self.lr,
                        'noise_std': config.get('noise', {}).get('target_noise_std_hu', 15.0),
                        'lambda_l1': lambda_l1,
                        'lambda_noise_residual': lambda_noise_residual,
                        'lambda_ssim': lambda_ssim,
                        'lambda_edge': lambda_edge,
                        'lambda_lowfreq': lambda_lowfreq,
                        'lambda_texture': lambda_texture,
                        'lambda_artifact': lambda_artifact,
                        'lambda_uniformity': lambda_uniformity,
                        'lambda_adversarial': self.lambda_adversarial,
                        'inpaint_trust': inpaint_trust,
                        'use_discriminator': self.use_discriminator,
                        'params': sum(p.numel() for p in self.model.parameters()),
                    },
                    resume="allow",
                )
                # Metric axis 설정 (epoch 기준 x축)
                wandb.define_metric("epoch")
                wandb.define_metric("train/*", step_metric="epoch")
                wandb.define_metric("val/*", step_metric="epoch")
                wandb.define_metric("lr", step_metric="epoch")
                self.logger.info("WandB initialized successfully")
            except Exception as e:
                self.logger.warning(f"WandB init failed: {e}")
                self.use_wandb = False

        self.logger.info("=" * 60)
        self.logger.info("MA-HybridNet Training Ready")
        self.logger.info(f"Epochs: {self.start_epoch} → {self.epochs}")
        self.logger.info("=" * 60)

    def _create_scheduler(self):
        """Warmup + Cosine Annealing scheduler."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                # Linear warmup
                return (epoch + 1) / self.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
                return max(self.min_lr / self.lr, 0.5 * (1 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def _load_fixed_sample(self):
        """Fixed sample 로드 (시각화용)."""
        from dataset import load_nifti_volume_autoaxis, center_crop, normalize_hu

        # 고정 환자 ID (config에서 읽거나 기본값)
        vis_cfg = self.config.get('ma_hybrid', {})
        fixed_patient_id = vis_cfg.get('fixed_sample_patient', '1728852')
        fixed_slice_idx = vis_cfg.get('fixed_sample_slice', 27)

        nifti_path = Path(self.nifti_root) / f"{fixed_patient_id}.nii"
        if not nifti_path.exists():
            nifti_path = Path(self.nifti_root) / f"{fixed_patient_id}.nii.gz"

        if not nifti_path.exists():
            self.logger.warning(f"[WARN] Fixed sample not found: {fixed_patient_id}")
            self.fixed_sample = None
            return

        vol = load_nifti_volume_autoaxis(str(nifti_path))

        if fixed_slice_idx >= vol.shape[0]:
            fixed_slice_idx = vol.shape[0] // 2

        slice_hu = vol[fixed_slice_idx].copy()

        # 90도 회전 + 좌우반전 (올바른 orientation)
        slice_hu = np.rot90(slice_hu, k=-1)
        slice_hu = np.fliplr(slice_hu)

        # Crop
        slice_hu = center_crop(slice_hu, self.crop_size)

        # Input basis = raw NC-CT
        input_basis_hu = slice_hu.copy()

        # Target: streak-corrected (processed) NC-CT if available, else raw
        target_hu = slice_hu.copy()
        if self.processed_root:
            proc_path = Path(self.processed_root) / f"{fixed_patient_id}.nii"
            if not proc_path.exists():
                proc_path = Path(self.processed_root) / f"{fixed_patient_id}.nii.gz"
            if proc_path.exists():
                proc_vol = load_nifti_volume_autoaxis(str(proc_path))
                proc_slice = proc_vol[fixed_slice_idx].copy()
                proc_slice = np.rot90(proc_slice, k=-1)
                proc_slice = np.fliplr(proc_slice)
                target_hu = center_crop(proc_slice, self.crop_size)
                self.logger.info(f"  Fixed sample target: streak-corrected (processed)")

        # Input (raw NC-CT + NPS noise)
        if self.noise_model is not None:
            input_hu = self.noise_model.add_noise(input_basis_hu)
        else:
            input_hu = input_basis_hu.copy()

        # Normalize
        input_norm = normalize_hu(input_hu, self.hu_min, self.hu_max)
        target_norm = normalize_hu(target_hu, self.hu_min, self.hu_max)

        # To tensor
        self.fixed_sample = {
            'input': torch.from_numpy(input_norm).unsqueeze(0).unsqueeze(0).float().to(self.device),
            'input_hu': torch.from_numpy(input_hu).unsqueeze(0).unsqueeze(0).float().to(self.device),
            'target': torch.from_numpy(target_norm).unsqueeze(0).unsqueeze(0).float().to(self.device),
        }

        self.logger.info(f"Fixed sample loaded: {fixed_patient_id}, slice {fixed_slice_idx}")

    def _load_checkpoint(self, ckpt_path: str):
        """체크포인트 로드."""
        self.logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.start_epoch = ckpt.get('epoch', 0) + 1
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))

        if 'scheduler' in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if 'scaler' in ckpt and self.scaler is not None:
            self.scaler.load_state_dict(ckpt['scaler'])

        # Discriminator checkpoint restore
        if 'discriminator' in ckpt and self.discriminator is not None:
            self.discriminator.load_state_dict(ckpt['discriminator'])
            self.logger.info("  Discriminator state restored")
        if 'd_optimizer' in ckpt and self.d_optimizer is not None:
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.logger.info("  D optimizer state restored")

        self.logger.info(f"Resumed from epoch {self.start_epoch}, best_val_loss: {self.best_val_loss:.6f}")

    def train_epoch(self, epoch: int):
        """한 epoch 학습 (Generator + Discriminator)."""
        self.model.train()
        if self.discriminator is not None:
            self.discriminator.train()

        total_loss = 0.0
        total_d_loss = 0.0
        total_g_adv_loss = 0.0
        loss_components = {}

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')

        for batch_idx, batch in enumerate(pbar):
            input_norm = batch['input'].to(self.device)      # (B, 1, H, W) [-1, 1]
            target_norm = batch['target'].to(self.device)    # (B, 1, H, W) [-1, 1]
            input_hu = batch['input_hu'].to(self.device)     # (B, 1, H, W) HU
            target_hu = batch['target_hu'].to(self.device)   # (B, 1, H, W) HU

            # Action-guided masks (optional, from pre-computed .npz)
            action_masks = None
            if 'mask_artifact' in batch:
                action_masks = {
                    'artifact': batch['mask_artifact'].to(self.device),
                    'fluid': batch['mask_fluid'].to(self.device),
                    'structure': batch['mask_structure'].to(self.device),
                }

            # Inpaint mask (|raw - processed| diff, from dataset)
            inpaint_mask = None
            if 'inpaint_mask' in batch:
                inpaint_mask = batch['inpaint_mask'].to(self.device)

            # === Generator forward ===
            pred, aux = self.model(input_norm, input_hu)

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                self.logger.warning(f"Batch {batch_idx}: Model output NaN/Inf, skipping")
                continue

            # === Discriminator update (if enabled) ===
            d_loss_val = 0.0
            g_adv_loss = torch.tensor(0.0, device=self.device)

            if self.discriminator is not None and self.ce_pool is not None:
                B = pred.shape[0]

                # Sobel gradients
                fake_grad = self.compute_sobel_2ch(pred.detach())
                ce_batch = self.ce_pool.sample(B, self.device)
                real_grad = self.compute_sobel_2ch(ce_batch)

                # D update: real→1, fake→0
                self.d_optimizer.zero_grad()
                d_real = self.discriminator(real_grad)
                d_fake = self.discriminator(fake_grad)

                d_loss_real = F.binary_cross_entropy_with_logits(
                    d_real, torch.ones_like(d_real))
                d_loss_fake = F.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake))
                d_loss = (d_loss_real + d_loss_fake) * 0.5

                # D loss NaN 방어: NaN 시 D 업데이트 스킵 (가중치 오염 방지)
                if torch.isnan(d_loss) or torch.isinf(d_loss):
                    self.logger.warning(f"Batch {batch_idx}: D loss NaN/Inf, skipping D update")
                    self.d_optimizer.zero_grad()
                else:
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.discriminator.parameters(), max_norm=self.d_grad_clip_norm)
                    self.d_optimizer.step()
                    d_loss_val = d_loss.item()

                # G adversarial loss: D(fake)→1
                fake_grad_g = self.compute_sobel_2ch(pred)
                d_fake_g = self.discriminator(fake_grad_g)
                g_adv_loss = F.binary_cross_entropy_with_logits(
                    d_fake_g, torch.ones_like(d_fake_g))

            # === Generator loss ===
            loss, loss_dict = self.criterion(
                pred, target_norm,
                input_noisy=input_norm,
                noise_pred=aux['noise_pred'],
                water_mask=aux['water_mask'],
                bone_mask=aux['bone_mask'],
                action_masks=action_masks,
                inpaint_mask=inpaint_mask,
            )

            # Add adversarial loss to generator
            loss = loss + self.lambda_adversarial * g_adv_loss
            loss_dict['g_adv'] = g_adv_loss.item()
            loss_dict['d_loss'] = d_loss_val

            # NaN/Inf 체크 (배치 스킵)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_losses = [k for k, v in loss_dict.items()
                              if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
                self.logger.warning(f"Batch {batch_idx}: NaN/Inf in {nan_losses}")
                continue

            # G backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            # Accumulate
            total_loss += loss.item()
            total_d_loss += d_loss_val
            total_g_adv_loss += g_adv_loss.item()
            for k, v in loss_dict.items():
                if k not in loss_components:
                    loss_components[k] = 0.0
                loss_components[k] += v

            # Progress bar
            pbar.set_postfix({'loss': loss.item(), 'd': d_loss_val})

        n_batches = len(self.train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}

        # Log
        loss_str = ' | '.join([f'{k}: {v:.4f}' for k, v in avg_components.items()
                               if k not in ('total', 'd_loss', 'g_adv')])
        extra = ""
        if self.discriminator is not None:
            extra = f" | d_loss: {total_d_loss/n_batches:.4f} | g_adv: {total_g_adv_loss/n_batches:.4f}"
        self.logger.info(f'Epoch [{epoch}/{self.epochs}] Train Loss: {avg_loss:.4f} | {loss_str}{extra}')

        self._last_train_components = avg_components
        return avg_loss

    def validate(self, epoch: int):
        """Validation."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_norm = batch['input'].to(self.device)
                target_norm = batch['target'].to(self.device)
                input_hu = batch['input_hu'].to(self.device)
                target_hu = batch['target_hu'].to(self.device)

                action_masks = None
                if 'mask_artifact' in batch:
                    action_masks = {
                        'artifact': batch['mask_artifact'].to(self.device),
                        'fluid': batch['mask_fluid'].to(self.device),
                        'structure': batch['mask_structure'].to(self.device),
                    }

                inpaint_mask = None
                if 'inpaint_mask' in batch:
                    inpaint_mask = batch['inpaint_mask'].to(self.device)

                pred, aux = self.model(input_norm, input_hu)
                loss, _ = self.criterion(
                    pred, target_norm,
                    input_noisy=input_norm,
                    noise_pred=aux['noise_pred'],
                    water_mask=aux['water_mask'],
                    bone_mask=aux['bone_mask'],
                    action_masks=action_masks,
                    inpaint_mask=inpaint_mask,
                )

                total_loss += loss.item()

                # Metrics
                psnr = compute_psnr(pred, target_norm, data_range=2.0)
                ssim = compute_ssim(pred, target_norm)
                total_psnr += psnr
                total_ssim += ssim

        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)

        self.logger.info(f"  [Val] Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")

        self._last_val_psnr = avg_psnr
        self._last_val_ssim = avg_ssim

        # Fixed sample visualization (매 10 에폭)
        if self.fixed_sample is not None and epoch % 10 == 0:
            self._save_fixed_sample(epoch, avg_psnr, avg_ssim)

        return avg_loss

    def _save_fixed_sample(self, epoch: int, psnr: float, ssim: float):
        """Fixed sample 시각화."""
        self.model.eval()
        with torch.no_grad():
            fixed_inp = self.fixed_sample['input']
            fixed_inp_hu = self.fixed_sample['input_hu']
            fixed_tgt = self.fixed_sample['target']

            fixed_out, aux = self.model(fixed_inp, fixed_inp_hu)

            # Save
            sample_path = self.samples_dir / f'ma_epoch_{epoch:04d}.png'
            save_sample_images(
                {'Input': fixed_inp,
                 'Output': fixed_out,
                 'Target': fixed_tgt},
                str(sample_path),
                epoch,
                metrics={'psnr': psnr, 'ssim': ssim},
                rotation=-2  # 180도 회전
            )

    def _update_curriculum_weights(self, epoch: int):
        """
        Curriculum Loss Scheduling.
        Phase 1 (0~transition): L1 높게(구조 안정화), artifact 낮게 시작 → 점진 증가
        Phase 2 (transition~):  config 설정값 유지
        """
        transition = self.config.get('ma_hybrid', {}).get('curriculum_epochs', 20)
        base_l1 = self.config.get('ma_hybrid', {}).get('lambda_l1', 2.0)
        base_art = self.config.get('ma_hybrid', {}).get('lambda_artifact', 1.5)

        if epoch < transition:
            alpha = epoch / transition
            self.criterion.lambda_l1 = base_l1 * 1.2       # 초반 구조 강조 +20%
            self.criterion.lambda_artifact = 0.3 + (base_art - 0.3) * alpha
        else:
            self.criterion.lambda_l1 = base_l1
            self.criterion.lambda_artifact = base_art

        if epoch % 10 == 0 or epoch < 5:
            self.logger.info(
                f"  [Curriculum] L1={self.criterion.lambda_l1:.2f}, "
                f"Artifact={self.criterion.lambda_artifact:.2f}")

    def run(self):
        """전체 학습 루프."""
        self.logger.info("Starting MA-HybridNet training...")

        for epoch in range(self.start_epoch, self.epochs):
            self._update_curriculum_weights(epoch)
            train_loss = self.train_epoch(epoch)

            val_loss = self.validate(epoch)

            # LR scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"  LR: {current_lr:.6f}")

            # WandB logging
            if self.use_wandb:
                try:
                    import wandb
                    log_dict = {
                        'epoch': epoch,
                        'train/loss': train_loss,
                        'val/loss': val_loss,
                        'val/psnr': self._last_val_psnr,
                        'val/ssim': self._last_val_ssim,
                        'lr': current_lr,
                        'curriculum/lambda_l1': self.criterion.lambda_l1,
                        'curriculum/lambda_artifact': self.criterion.lambda_artifact,
                    }
                    # 개별 loss 항목
                    if hasattr(self, '_last_train_components'):
                        for k, v in self._last_train_components.items():
                            if k != 'total':
                                log_dict[f'train/{k}'] = v
                    # 샘플 이미지 (10 에폭마다)
                    if epoch % 10 == 0:
                        sample_path = self.samples_dir / f'ma_epoch_{epoch:04d}.png'
                        if sample_path.exists():
                            log_dict['sample'] = wandb.Image(str(sample_path))
                    wandb.log(log_dict)
                    self.logger.info(
                        f"  [WandB] logged: train/loss={train_loss:.4f}, "
                        f"val/loss={val_loss:.4f}, val/psnr={self._last_val_psnr:.2f}"
                    )
                except Exception as e:
                    self.logger.warning(f"WandB log failed: {e}")

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or epoch == self.epochs - 1:
                ckpt_path = self.weights_dir / f'ckpt_epoch_{epoch:04d}.pth'
                ckpt_data = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'scaler': self.scaler.state_dict() if self.scaler else None,
                    'val_loss': val_loss,
                    'best_val_loss': self.best_val_loss,
                }
                if self.discriminator is not None:
                    ckpt_data['discriminator'] = self.discriminator.state_dict()
                    ckpt_data['d_optimizer'] = self.d_optimizer.state_dict()
                save_checkpoint(ckpt_data, str(ckpt_path))
                self.logger.info(f"  Checkpoint saved: {ckpt_path.name}")

            # Best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0

                best_path = self.weights_dir / 'best_model.pth'
                best_data = {
                    'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }
                if self.discriminator is not None:
                    best_data['discriminator'] = self.discriminator.state_dict()
                    best_data['d_optimizer'] = self.d_optimizer.state_dict()
                save_checkpoint(best_data, str(best_path))
                self.logger.info(f"  >> Best model saved (val_loss: {val_loss:.6f})")
            else:
                self.epochs_no_improve += 1

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                self.logger.info(f"Early stopping triggered (patience={self.patience})")
                break

        self.logger.info("MA-HybridNet training completed!")
        self.logger.info(f"Best val_loss: {self.best_val_loss:.6f}")

        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


# ============================================================
# 메인 디스패치
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description='MA-HybridNet Training Pipeline')
    p.add_argument('--config', type=str, default='config_ma_hybrid.yaml',
                   help='YAML 설정 파일 경로')
    p.add_argument('--mode', type=str, default=None,
                   choices=['ma_hybrid'],
                   help='실행 모드')
    p.add_argument('--resume', type=str, default=None,
                   help='체크포인트 경로 (학습 재개)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # YAML 설정 로드
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   config_path)

    cfg = load_config(config_path)

    # --resume CLI 인자 → config에 반영
    if args.resume:
        if 'ma_hybrid' in cfg:
            cfg['ma_hybrid']['resume_ckpt'] = args.resume

    # 모드 결정 (CLI > config)
    mode = args.mode or cfg.get('mode', 'ma_hybrid')
    print(f"[Mode] {mode}")

    if mode == 'ma_hybrid':
        trainer = MAHybridNetTrainer(cfg)
        trainer.run()

    else:
        print(f"[ERROR] Unknown mode: {mode}")
        sys.exit(1)
