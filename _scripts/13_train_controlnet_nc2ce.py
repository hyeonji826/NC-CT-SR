# -*- coding: utf-8 -*-
"""
NC to CE Style Transfer - Stable Diffusion + ControlNet + LoRA
NC를 구조 가이드로 사용하여 CE 스타일 생성
Imperfect pairing 문제 해결, End-to-End 학습
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch.utils._pytree.*')

import os
import argparse
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    DDIMScheduler,
    ControlNetModel
)
from diffusers.models.controlnet import ControlNetOutput
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer


# ============================================================
# Dataset: NC (control) + CE (target)
# ============================================================
class NCCEControlNetDataset(Dataset):
    """NC를 control로, CE를 target으로 사용하는 데이터셋"""
    
    def __init__(self, root, pairs_csv, split='train', image_size=512, num_slices=5):
        self.root = Path(root)
        self.image_size = image_size
        self.num_slices = num_slices
        
        # pairs.csv 로드
        df = pd.read_csv(self.root / pairs_csv)
        
        # Train/Val/Test split (8:1:1)
        n_total = len(df)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        if split == 'train':
            df = df.iloc[:n_train]
        elif split == 'val':
            df = df.iloc[n_train:n_train+n_val]
        else:  # test
            df = df.iloc[n_train+n_val:]
        
        df = df.reset_index(drop=True)
        
        self.pairs = []
        for _, row in df.iterrows():
            nc_path = Path(row['input_nc_norm'])
            ce_path = Path(row['target_ce_norm'])
            
            if nc_path.exists() and ce_path.exists():
                self.pairs.append({
                    'nc_path': nc_path,
                    'ce_path': ce_path,
                    'patient_id': row['id7']
                })
        
        print(f"[{split}] Loaded {len(self.pairs)} NC-CE pairs")
    
    def __len__(self):
        return len(self.pairs) * self.num_slices
    
    def __getitem__(self, idx):
        pair_idx = idx // self.num_slices
        slice_offset = idx % self.num_slices
        
        pair = self.pairs[pair_idx]
        
        # Load volumes
        nc_img = sitk.ReadImage(str(pair['nc_path']))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        ce_img = sitk.ReadImage(str(pair['ce_path']))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        
        # Select corresponding slices from middle region
        D_nc = nc_arr.shape[0]
        D_ce = ce_arr.shape[0]
        
        # NC slice
        start_nc = D_nc // 4
        end_nc = 3 * D_nc // 4
        slice_idx_nc = start_nc + (end_nc - start_nc) * slice_offset // self.num_slices
        slice_idx_nc = np.clip(slice_idx_nc, 0, D_nc - 1)
        nc_slice = nc_arr[slice_idx_nc]
        
        # CE slice (corresponding position)
        slice_idx_ce = int(slice_idx_nc * D_ce / D_nc)
        slice_idx_ce = np.clip(slice_idx_ce, 0, D_ce - 1)
        ce_slice = ce_arr[slice_idx_ce]
        
        # Resize CE to match NC
        if ce_slice.shape != nc_slice.shape:
            from skimage.transform import resize
            ce_slice = resize(ce_slice, nc_slice.shape, 
                            order=1, preserve_range=True, anti_aliasing=True)
        
        # Convert to [-1, 1] range
        nc_slice = nc_slice * 2.0 - 1.0
        ce_slice = ce_slice * 2.0 - 1.0
        
        # To tensor
        nc_tensor = torch.from_numpy(nc_slice).unsqueeze(0)  # [1, H, W]
        ce_tensor = torch.from_numpy(ce_slice).unsqueeze(0)  # [1, H, W]
        
        # Resize
        if nc_tensor.shape[1] != self.image_size or nc_tensor.shape[2] != self.image_size:
            nc_tensor = transforms.functional.resize(
                nc_tensor, (self.image_size, self.image_size), antialias=True
            )
            ce_tensor = transforms.functional.resize(
                ce_tensor, (self.image_size, self.image_size), antialias=True
            )
        
        # Prompt
        prompt = "high quality contrast-enhanced CT scan with clear tissue boundaries and enhanced blood vessels"
        
        return {
            'control_image': nc_tensor,  # NC as control
            'target_image': ce_tensor,   # CE as target
            'prompt': prompt
        }


# ============================================================
# Collate function
# ============================================================
def collate_fn(examples):
    control_images = torch.stack([ex['control_image'] for ex in examples])
    target_images = torch.stack([ex['target_image'] for ex in examples])
    
    # Repeat to 3 channels
    control_images = control_images.repeat(1, 3, 1, 1)
    target_images = target_images.repeat(1, 3, 1, 1)
    
    prompts = [ex['prompt'] for ex in examples]
    
    return {
        'control_images': control_images,
        'target_images': target_images,
        'prompts': prompts
    }


# ============================================================
# Trainer
# ============================================================
class ControlNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Directories
        self.exp_dir = Path(args.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'samples').mkdir(exist_ok=True)
        
        print("Loading Stable Diffusion + ControlNet components...")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # Text encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float32
        ).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        # ControlNet - 처음부터 학습
        print("Initializing ControlNet from scratch...")
        self.controlnet = ControlNetModel.from_unet(
            UNet2DConditionModel.from_pretrained(
                model_id, subfolder="unet", torch_dtype=torch.float32
            )
        ).to(self.device)
        
        # UNet with LoRA
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float32
        )
        
        # LoRA for UNet
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none"
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.to(self.device)
        
        # Trainable parameters
        controlnet_params = sum(p.numel() for p in self.controlnet.parameters())
        unet_trainable = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        print(f"ControlNet params: {controlnet_params:,}")
        print(f"UNet trainable (LoRA): {unet_trainable:,}")
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        # Optimizer - ControlNet + UNet LoRA
        self.optimizer = torch.optim.AdamW(
            list(self.controlnet.parameters()) + list(self.unet.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # LR Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )
        
        # Datasets
        self.train_dataset = NCCEControlNetDataset(
            args.root, args.pairs_csv, split='train',
            image_size=args.image_size, num_slices=args.num_slices
        )
        self.val_dataset = NCCEControlNetDataset(
            args.root, args.pairs_csv, split='val',
            image_size=args.image_size, num_slices=2
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def encode_prompt(self, prompts):
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        return prompt_embeds
    
    def train_epoch(self, epoch):
        self.controlnet.train()
        self.unet.train()
        
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch in pbar:
            control_images = batch['control_images'].to(self.device, dtype=torch.float32)
            target_images = batch['target_images'].to(self.device, dtype=torch.float32)
            prompts = batch['prompts']
            
            # Encode control images (NC)
            with torch.no_grad():
                control_latents = self.vae.encode(control_images).latent_dist.sample()
                control_latents = control_latents * self.vae.config.scaling_factor
            
            # Encode target images (CE)
            with torch.no_grad():
                target_latents = self.vae.encode(target_images).latent_dist.sample()
                target_latents = target_latents * self.vae.config.scaling_factor
            
            # Text embeddings
            encoder_hidden_states = self.encode_prompt(prompts)
            
            # Add noise to target
            noise = torch.randn_like(target_latents)
            bsz = target_latents.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=target_latents.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
            
            # ControlNet forward - NC를 control로 사용
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control_latents,  # NC latent as control
                return_dict=False
            )
            
            # UNet forward with ControlNet conditioning
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample
            
            # Loss
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Skipping batch with NaN/Inf loss")
                continue
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.controlnet.parameters()) + list(self.unet.parameters()), 
                1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(valid_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        self.controlnet.eval()
        self.unet.eval()
        
        total_loss = 0
        valid_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            control_images = batch['control_images'].to(self.device, dtype=torch.float32)
            target_images = batch['target_images'].to(self.device, dtype=torch.float32)
            prompts = batch['prompts']
            
            control_latents = self.vae.encode(control_images).latent_dist.sample()
            control_latents = control_latents * self.vae.config.scaling_factor
            
            target_latents = self.vae.encode(target_images).latent_dist.sample()
            target_latents = target_latents * self.vae.config.scaling_factor
            
            encoder_hidden_states = self.encode_prompt(prompts)
            
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (target_latents.shape[0],), device=target_latents.device
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
            
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                noisy_latents, timesteps, encoder_hidden_states,
                controlnet_cond=control_latents, return_dict=False
            )
            
            model_pred = self.unet(
                noisy_latents, timesteps, encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample
            
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                valid_batches += 1
        
        avg_loss = total_loss / max(valid_batches, 1)
        self.val_losses.append(avg_loss)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best')
            
            # Generate sample
            self.generate_sample(epoch, batch)
            
            print(f"✓ Best model saved (val_loss: {avg_loss:.4f})")
        
        return avg_loss
    
    @torch.no_grad()
    def generate_sample(self, epoch, batch):
        """샘플 생성 - NC control로 CE style 생성"""
        self.controlnet.eval()
        self.unet.eval()
        
        # 첫 번째 샘플만 사용
        control_image = batch['control_images'][:1].to(self.device, dtype=torch.float32)
        target_image = batch['target_images'][:1].to(self.device, dtype=torch.float32)
        prompt = batch['prompts'][:1]
        
        # Encode control
        control_latents = self.vae.encode(control_image).latent_dist.sample()
        control_latents = control_latents * self.vae.config.scaling_factor
        
        # Text embedding
        encoder_hidden_states = self.encode_prompt(prompt)
        
        # DDIM scheduler for sampling
        ddim_scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler"
        )
        ddim_scheduler.set_timesteps(50)
        
        # Random latent
        latents = torch.randn(
            1, 4, self.args.image_size // 8, self.args.image_size // 8,
            device=self.device, dtype=torch.float32
        )
        
        # Denoising loop
        for t in tqdm(ddim_scheduler.timesteps, desc="Generating", leave=False):
            # ControlNet
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latents, t, encoder_hidden_states,
                controlnet_cond=control_latents, return_dict=False
            )
            
            # UNet prediction
            noise_pred = self.unet(
                latents, t, encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample
            
            latents = ddim_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        latents = 1 / self.vae.config.scaling_factor * latents
        generated_image = self.vae.decode(latents).sample
        
        # Save comparison
        import matplotlib.pyplot as plt
        
        # Denormalize
        control_np = (control_image[0, 0].cpu().numpy() / 2 + 0.5).clip(0, 1)
        generated_np = (generated_image[0, 0].cpu().numpy() / 2 + 0.5).clip(0, 1)
        target_np = (target_image[0, 0].cpu().numpy() / 2 + 0.5).clip(0, 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(control_np, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Input NC (Control)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(generated_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Generated CE-style', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(target_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Target CE', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Best Model - Epoch {epoch+1}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.exp_dir / 'samples' / f'best_epoch_{epoch+1:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Sample saved: {save_path.name}")
    
    def save_checkpoint(self, name):
        """ControlNet + UNet LoRA 저장"""
        save_dir = self.exp_dir / 'checkpoints' / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # ControlNet
        self.controlnet.save_pretrained(save_dir / 'controlnet')
        
        # UNet LoRA
        self.unet.save_pretrained(save_dir / 'unet_lora')
        
        # Training state
        torch.save({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_dir / 'training_state.pth')
    
    def train(self):
        print(f"\n{'='*80}")
        print("NC to CE Style Transfer with ControlNet + LoRA")
        print(f"{'='*80}\n")
        
        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.lr_scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1:03d}')
        
        print("\n✓ Training complete!")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='NC to CE with ControlNet')
    
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--exp-dir', default='Outputs/experiments/controlnet_nc2ce')
    
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--num-slices', type=int, default=5)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Train
    trainer = ControlNetTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()