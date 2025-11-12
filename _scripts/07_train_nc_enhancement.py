# -*- coding: utf-8 -*-
"""
NC CT Quality Enhancement - Paired Learning
NC를 입력으로, CE를 타겟으로 하는 화질 개선 모델
목표: 노이즈 제거 + 간암 가시성 향상
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
# Dataset: NC-CE Paired
# ============================================================
class NCCEPairedDataset(Dataset):
    """NC-CE 쌍 데이터셋"""
    
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
                    'ce_path': ce_path
                })
        
        print(f"[{split}] Loaded {len(self.pairs)} NC-CE pairs")
    
    def __len__(self):
        return len(self.pairs) * self.num_slices
    
    def __getitem__(self, idx):
        pair_idx = idx // self.num_slices
        slice_offset = idx % self.num_slices
        
        pair = self.pairs[pair_idx]
        
        # NC 볼륨 로드
        nc_img = sitk.ReadImage(str(pair['nc_path']))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        # CE 볼륨 로드
        ce_img = sitk.ReadImage(str(pair['ce_path']))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        
        # 같은 위치의 슬라이스 선택
        D_nc, H, W = nc_arr.shape
        D_ce = ce_arr.shape[0]
        
        # 중간 부분에서 슬라이스 선택
        start_idx = min(D_nc, D_ce) // 4
        end_idx = 3 * min(D_nc, D_ce) // 4
        slice_idx = start_idx + (end_idx - start_idx) * slice_offset // self.num_slices
        slice_idx = np.clip(slice_idx, 0, min(D_nc, D_ce) - 1)
        
        nc_slice = nc_arr[min(slice_idx, D_nc-1)]
        ce_slice = ce_arr[min(slice_idx, D_ce-1)]
        
        # 데이터가 이미 [0, 1]로 정규화되어 있음
        # [-1, 1] 범위로만 변환
        nc_slice = nc_slice * 2.0 - 1.0
        ce_slice = ce_slice * 2.0 - 1.0
        
        # Tensor로 변환
        nc_tensor = torch.from_numpy(nc_slice).unsqueeze(0)  # [1, H, W]
        ce_tensor = torch.from_numpy(ce_slice).unsqueeze(0)
        
        # Resize
        if nc_tensor.shape[1] != self.image_size or nc_tensor.shape[2] != self.image_size:
            nc_tensor = transforms.functional.resize(
                nc_tensor, 
                (self.image_size, self.image_size), 
                antialias=True
            )
            ce_tensor = transforms.functional.resize(
                ce_tensor,
                (self.image_size, self.image_size),
                antialias=True
            )
        
        # Prompt
        prompt = "high quality CT scan, denoised medical image, enhanced liver contrast"
        
        return {
            'nc': nc_tensor,  # Input
            'ce': ce_tensor,  # Target
            'prompt': prompt
        }


# ============================================================
# Collate function
# ============================================================
def collate_fn(examples):
    nc_values = torch.stack([ex['nc'] for ex in examples])
    ce_values = torch.stack([ex['ce'] for ex in examples])
    
    # RGB로 변환
    nc_values = nc_values.repeat(1, 3, 1, 1)
    ce_values = ce_values.repeat(1, 3, 1, 1)
    
    prompts = [ex['prompt'] for ex in examples]
    
    return {
        'nc': nc_values,
        'ce': ce_values,
        'prompts': prompts
    }


# ============================================================
# Trainer
# ============================================================
class NCEnhancementTrainer:
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
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
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
        self.train_dataset = NCCEPairedDataset(
            args.root, args.pairs_csv, split='train',
            image_size=args.image_size, num_slices=args.num_slices
        )
        self.val_dataset = NCCEPairedDataset(
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
            nc_images = batch['nc'].to(self.device, dtype=torch.float32)
            ce_images = batch['ce'].to(self.device, dtype=torch.float32)
            prompts = batch['prompts']
            
            # CE를 타겟으로 학습
            with torch.no_grad():
                ce_latents = self.vae.encode(ce_images).latent_dist.sample()
                ce_latents = ce_latents * self.vae.config.scaling_factor
            
            encoder_hidden_states = self.encode_prompt(prompts)
            
            noise = torch.randn_like(ce_latents)
            bsz = ce_latents.shape[0]
            
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=ce_latents.device
            ).long()
            
            noisy_latents = self.noise_scheduler.add_noise(ce_latents, noise, timesteps)
            
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
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
            nc_images = batch['nc'].to(self.device, dtype=torch.float32)
            ce_images = batch['ce'].to(self.device, dtype=torch.float32)
            prompts = batch['prompts']
            
            ce_latents = self.vae.encode(ce_images).latent_dist.sample()
            ce_latents = ce_latents * self.vae.config.scaling_factor
            encoder_hidden_states = self.encode_prompt(prompts)
            
            noise = torch.randn_like(ce_latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (ce_latents.shape[0],), device=ce_latents.device
            ).long()
            noisy_latents = self.noise_scheduler.add_noise(ce_latents, noise, timesteps)
            
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
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
        print("Starting NC CT Quality Enhancement (Paired Learning)")
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


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='NC CT Quality Enhancement')
    
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--exp-dir', default='Outputs/experiments/nc_enhancement')
    
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--num-slices', type=int, default=5)
    
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--num-workers', type=int, default=4)
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print(f"GPU Configuration:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"{'='*80}\n")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    trainer = NCEnhancementTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()