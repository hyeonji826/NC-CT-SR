# -*- coding: utf-8 -*-
"""
CE CT 이미지 스타일 학습 - Stable Diffusion + LoRA
CE 이미지만 사용해서 CT contrast enhancement 스타일을 학습합니다.
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
# Dataset: CE CT 이미지만 로드
# ============================================================
class CECTDataset(Dataset):
    """CE CT 이미지만 사용하는 데이터셋"""
    
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
        
        self.ce_paths = []
        for _, row in df.iterrows():
            # target_ce_norm 경로 사용
            ce_path = Path(row['target_ce_norm'])
            if ce_path.exists():
                self.ce_paths.append(ce_path)
        
        print(f"[{split}] Loaded {len(self.ce_paths)} CE CT volumes")
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.ce_paths) * self.num_slices  # 볼륨당 여러 슬라이스
    
    def __getitem__(self, idx):
        vol_idx = idx // self.num_slices
        slice_offset = idx % self.num_slices
        
        # CE 볼륨 로드
        ce_img = sitk.ReadImage(str(self.ce_paths[vol_idx]))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        
        # 중간 부분에서 슬라이스 선택
        D, H, W = ce_arr.shape
        start_idx = D // 4
        end_idx = 3 * D // 4
        slice_idx = start_idx + (end_idx - start_idx) * slice_offset // self.num_slices
        slice_idx = np.clip(slice_idx, 0, D - 1)
        
        ce_slice = ce_arr[slice_idx]
        
        # HU clipping: [-160, 240] (soft tissue window)
        ce_slice = np.clip(ce_slice, -160, 240)
        ce_slice = (ce_slice + 160) / 400.0  # [0, 1]
        
        # Tensor로 변환
        ce_tensor = torch.from_numpy(ce_slice).unsqueeze(0)  # [1, H, W]
        ce_tensor = self.transform(ce_tensor)
        
        # Prompt: CE CT 스타일 설명
        prompt = "contrast-enhanced CT scan, medical imaging, enhanced blood vessels"
        
        return {
            'pixel_values': ce_tensor,  # [1, H, W]
            'prompt': prompt
        }


# ============================================================
# Collate function
# ============================================================
def collate_fn(examples):
    pixel_values = torch.stack([ex['pixel_values'] for ex in examples])
    pixel_values = pixel_values.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]
    prompts = [ex['prompt'] for ex in examples]
    return {'pixel_values': pixel_values, 'prompts': prompts}


# ============================================================
# Trainer
# ============================================================
class SDLoRATrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Directories
        self.exp_dir = Path(args.exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.exp_dir / 'samples').mkdir(exist_ok=True)
        
        # Load pretrained Stable Diffusion components
        print("Loading Stable Diffusion components...")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        # Use float32 to prevent NaN issues with CT images
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
        
        # LoRA Config
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none"
        )
        self.unet = get_peft_model(self.unet, lora_config)
        self.unet.to(self.device)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.unet.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.unet.parameters())
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        # Noise scheduler
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
        
        # LR Scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=1e-6
        )
        
        # Datasets
        self.train_dataset = CECTDataset(
            args.root, args.pairs_csv, split='train',
            image_size=args.image_size, num_slices=args.num_slices
        )
        self.val_dataset = CECTDataset(
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
        """텍스트를 임베딩으로 변환"""
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
            pixel_values = batch['pixel_values'].to(self.device, dtype=torch.float32)
            prompts = batch['prompts']
            
            # Encode images to latent space
            with torch.no_grad():
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            
            # Encode prompts
            encoder_hidden_states = self.encode_prompt(prompts)
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=latents.device
            ).long()
            
            # Add noise to latents
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # Loss with safety check
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Skipping batch with NaN/Inf loss")
                continue
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (더 강하게)
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
            pixel_values = batch['pixel_values'].to(self.device, dtype=torch.float32)
            prompts = batch['prompts']
            
            # Encode
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            encoder_hidden_states = self.encode_prompt(prompts)
            
            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=latents.device
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = F.mse_loss(model_pred, noise, reduction="mean")
            
            # Check for NaN
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                valid_batches += 1
        
        avg_loss = total_loss / max(valid_batches, 1)
        self.val_losses.append(avg_loss)
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint('best.pth')
            print(f"✓ Best model saved (val_loss: {avg_loss:.4f})")
        
        return avg_loss
    
    @torch.no_grad()
    def sample_images(self, epoch):
        """샘플 생성 (텍스트 프롬프트만으로)"""
        self.unet.eval()
        
        # DDIM scheduler for faster sampling
        ddim_scheduler = DDIMScheduler.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", subfolder="scheduler"
        )
        ddim_scheduler.set_timesteps(50)
        
        prompt = "contrast-enhanced CT scan, medical imaging, enhanced blood vessels"
        encoder_hidden_states = self.encode_prompt([prompt])
        
        # Random latent
        latents = torch.randn(
            1, 4, self.args.image_size // 8, self.args.image_size // 8,
            device=self.device, dtype=torch.float32
        )
        
        # Denoising loop
        for t in tqdm(ddim_scheduler.timesteps, desc="Sampling", leave=False):
            latent_model_input = latents
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            latents = ddim_scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        
        # Save
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        
        save_path = self.exp_dir / 'samples' / f'epoch_{epoch:03d}.png'
        Image.fromarray(image).save(save_path)
        print(f"Sample saved: {save_path}")
    
    def save_checkpoint(self, filename):
        """LoRA weights만 저장"""
        save_path = self.exp_dir / 'checkpoints' / filename
        
        # LoRA adapter만 저장
        self.unet.save_pretrained(save_path)
        
        # Training state
        torch.save({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, save_path / 'training_state.pth')
    
    def train(self):
        print(f"\n{'='*80}")
        print("Starting CE CT Style Fine-tuning with Stable Diffusion + LoRA")
        print(f"{'='*80}\n")
        
        for epoch in range(self.args.epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # LR step
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {current_lr:.2e}")
            
            # Sample images every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.sample_images(epoch + 1)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1:03d}.pth')
        
        print("\n✓ Training complete!")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='CE CT Style Fine-tuning with SD + LoRA')
    
    # Data
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--exp-dir', default='Outputs/experiments/sd_lora_ce_style')
    
    # Model
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank (낮을수록 가벼움)')
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--num-slices', type=int, default=5, help='볼륨당 추출할 슬라이스 개수')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print(f"GPU Configuration:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"{'='*80}\n")
    else:
        print("WARNING: CUDA not available, using CPU")
    
    # Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Train
    trainer = SDLoRATrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
