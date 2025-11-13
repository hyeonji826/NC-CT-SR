# -*- coding: utf-8 -*-
"""
NC to CE Style Transfer - Stable Diffusion + LoRA
Denoised NC를 input으로 받아 CE 스타일로 변환
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
    DDIMScheduler
)
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer

# ============================================================
# Dataset: Denoised NC → CE paired data
# ============================================================
class NC2CEDataset(Dataset):
    """Denoised NC를 input으로, CE를 target으로 하는 데이터셋"""
    
    def __init__(self, root, pairs_csv, denoised_nc_dir, split='train', image_size=512, num_slices=5):
        self.root = Path(root)
        self.denoised_nc_dir = Path(denoised_nc_dir)
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
            patient_id = row['id7']
            ce_path = Path(row['target_ce_norm'])
            
            # Denoised NC 경로
            denoised_nc_path = self.denoised_nc_dir / f"{patient_id}_denoised.nii.gz"
            
            if ce_path.exists() and denoised_nc_path.exists():
                self.pairs.append({
                    'nc_path': denoised_nc_path,
                    'ce_path': ce_path,
                    'patient_id': patient_id
                })
        
        print(f"[{split}] Loaded {len(self.pairs)} NC-CE pairs")
    
    def __len__(self):
        return len(self.pairs) * self.num_slices
    
    def __getitem__(self, idx):
        pair_idx = idx // self.num_slices
        slice_offset = idx % self.num_slices
        
        pair = self.pairs[pair_idx]
        
        # Load NC and CE volumes
        nc_img = sitk.ReadImage(str(pair['nc_path']))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        ce_img = sitk.ReadImage(str(pair['ce_path']))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        
        # Resize CE to match NC if needed
        if ce_arr.shape != nc_arr.shape:
            from skimage.transform import resize
            ce_arr = resize(ce_arr, nc_arr.shape, order=1, preserve_range=True, anti_aliasing=True)
        
        # Select slice from middle region
        D = nc_arr.shape[0]
        start_idx = D // 4
        end_idx = 3 * D // 4
        slice_idx = start_idx + (end_idx - start_idx) * slice_offset // self.num_slices
        slice_idx = np.clip(slice_idx, 0, D - 1)
        
        nc_slice = nc_arr[slice_idx]
        ce_slice = ce_arr[slice_idx]
        
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
        prompt = "contrast-enhanced CT scan with improved tissue contrast and enhanced blood vessels"
        
        return {
            'input_nc': nc_tensor,      # Input: Denoised NC
            'target_ce': ce_tensor,     # Target: CE
            'prompt': prompt
        }


# ============================================================
# Collate function
# ============================================================
def collate_fn(examples):
    input_nc = torch.stack([ex['input_nc'] for ex in examples])
    target_ce = torch.stack([ex['target_ce'] for ex in examples])
    
    # Repeat to 3 channels
    input_nc = input_nc.repeat(1, 3, 1, 1)  # [B, 3, H, W]
    target_ce = target_ce.repeat(1, 3, 1, 1)
    
    prompts = [ex['prompt'] for ex in examples]
    
    return {
        'input_nc': input_nc,
        'target_ce': target_ce,
        'prompts': prompts
    }


# ============================================================
# Trainer
# ============================================================
class NC2CETrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Directories
        self.exp_dir = Path(args.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'samples').mkdir(exist_ok=True)
        
        # Load pretrained Stable Diffusion
        print("Loading Stable Diffusion components...")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float32
        ).to(self.device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        
        # UNet with LoRA
        self.unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float32
        )
        
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none"
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.to(self.device)
        
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )
        
        # Datasets
        self.train_dataset = NC2CEDataset(
            args.root, args.pairs_csv, args.denoised_nc_dir,
            split='train', image_size=args.image_size, num_slices=args.num_slices
        )
        self.val_dataset = NC2CEDataset(
            args.root, args.pairs_csv, args.denoised_nc_dir,
            split='val', image_size=args.image_size, num_slices=2
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
        self.unet.train()
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch in pbar:
            input_nc = batch['input_nc'].to(self.device, dtype=torch.float32)
            target_ce = batch['target_ce'].to(self.device, dtype=torch.float32)
            prompts = batch['prompts']
            
            # Encode NC (as conditioning) and CE (as target)
            with torch.no_grad():
                # Input NC를 latent space로
                nc_latents = self.vae.encode(input_nc).latent_dist.sample()
                nc_latents = nc_latents * self.vae.config.scaling_factor
                
                # Target CE를 latent space로
                ce_latents = self.vae.encode(target_ce).latent_dist.sample()
                ce_latents = ce_latents * self.vae.config.scaling_factor
            
            # Prompt encoding
            encoder_hidden_states = self.encode_prompt(prompts)
            
            # Add noise to CE latents (target)
            noise = torch.randn_like(ce_latents)
            bsz = ce_latents.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=ce_latents.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(ce_latents, noise, timesteps)
            
            # Concatenate NC latents as additional conditioning
            # (간단한 방법: noisy CE latents와 NC latents를 blend)
            # 더 나은 방법은 ControlNet이지만, 간단하게 구현
            conditioning_scale = 0.3
            conditioned_latents = noisy_latents + conditioning_scale * nc_latents
            
            # Predict noise
            model_pred = self.unet(
                conditioned_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # Loss
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Skipping batch with NaN/Inf loss")
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / max(valid_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        self.unet.eval()
        total_loss = 0
        valid_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_nc = batch['input_nc'].to(self.device, dtype=torch.float32)
            target_ce = batch['target_ce'].to(self.device, dtype=torch.float32)
            prompts = batch['prompts']
            
            nc_latents = self.vae.encode(input_nc).latent_dist.sample()
            nc_latents = nc_latents * self.vae.config.scaling_factor
            
            ce_latents = self.vae.encode(target_ce).latent_dist.sample()
            ce_latents = ce_latents * self.vae.config.scaling_factor
            
            encoder_hidden_states = self.encode_prompt(prompts)
            
            noise = torch.randn_like(ce_latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (ce_latents.shape[0],), device=ce_latents.device
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(ce_latents, noise, timesteps)
            
            conditioning_scale = 0.3
            conditioned_latents = noisy_latents + conditioning_scale * nc_latents
            
            model_pred = self.unet(conditioned_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                valid_batches += 1
        
        avg_loss = total_loss / max(valid_batches, 1)
        self.val_losses.append(avg_loss)
        
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best.pth')
            print(f"✓ Best model saved (val_loss: {avg_loss:.4f})")
        
        return avg_loss
    
    def save_checkpoint(self, filename):
        save_path = self.exp_dir / 'checkpoints' / filename
        self.unet.save_pretrained(save_path)
        torch.save({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_path / 'training_state.pth')
    
    def train(self):
        print(f"\n{'='*80}")
        print("Starting NC to CE Style Transfer Training")
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
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1:03d}.pth')
        
        print("\n✓ Training complete!")


def main():
    parser = argparse.ArgumentParser(description='NC to CE Style Transfer with SD + LoRA')
    
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--denoised-nc-dir', default='Data/denoised_nc',
                       help='Directory containing denoised NC volumes')
    parser.add_argument('--exp-dir', default='Outputs/experiments/sd_nc2ce_style')
    
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--num-slices', type=int, default=5)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    trainer = NC2CETrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()