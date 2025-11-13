#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk
import cv2
import random
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


class StructureLoss(nn.Module):
    def __init__(self, hu_min=-30, hu_max=150):
        super().__init__()
        self.hu_min = hu_min
        self.hu_max = hu_max
    
    def get_organ_mask(self, ct_image):
        hu = ct_image * 1000
        mask = (hu > self.hu_min) & (hu < self.hu_max)
        return mask.float()
    
    def dice_loss(self, pred_mask, target_mask):
        smooth = 1e-5
        intersection = (pred_mask * target_mask).sum(dim=(2, 3))
        union = pred_mask.sum(dim=(2, 3)) + target_mask.sum(dim=(2, 3))
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, pred, target):
        pred_mask = self.get_organ_mask(pred)
        target_mask = self.get_organ_mask(target)
        return self.dice_loss(pred_mask, target_mask)


def extract_edges(img):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    
    edge_x = F.conv2d(img, sobel_x, padding=1)
    edge_y = F.conv2d(img, sobel_y, padding=1)
    edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8) * 2 - 1
    return edge


class NCCEPairDataset(Dataset):
    def __init__(self, csv_path, image_size=512, augment=True):
        self.pairs = pd.read_csv(csv_path)
        self.image_size = image_size
        self.augment = augment
        
    def __len__(self):
        return len(self.pairs)
    
    def load_slice(self, nifti_path, slice_idx):
        img = sitk.ReadImage(str(nifti_path))
        arr = sitk.GetArrayFromImage(img)[slice_idx]
        arr = np.clip((arr + 1000) / 1000, -1, 1)
        return arr
    
    def augment_pair(self, nc, ce):
        if not self.augment:
            return nc, ce
        
        if random.random() > 0.5:
            nc = np.fliplr(nc)
            ce = np.fliplr(ce)
        
        if random.random() > 0.5:
            nc = np.flipud(nc)
            ce = np.flipud(ce)
        
        k = random.randint(0, 3)
        if k > 0:
            nc = np.rot90(nc, k)
            ce = np.rot90(ce, k)
        
        if random.random() > 0.5:
            scale = random.uniform(0.95, 1.05)
            shift = random.uniform(-0.05, 0.05)
            nc = np.clip(nc * scale + shift, -1, 1)
            ce = np.clip(ce * scale + shift, -1, 1)
        
        return nc, ce
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        nc = self.load_slice(row['nc_path'], row['nc_slice_idx'])
        ce = self.load_slice(row['ce_path'], row['ce_slice_idx'])
        
        nc, ce = self.augment_pair(nc, ce)
        
        nc = cv2.resize(nc, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        ce = cv2.resize(ce, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        
        nc = torch.from_numpy(nc).float().unsqueeze(0)
        ce = torch.from_numpy(ce).float().unsqueeze(0)
        
        edge = extract_edges(nc.unsqueeze(0)).squeeze(0)
        
        nc_rgb = nc.repeat(3, 1, 1)
        ce_rgb = ce.repeat(3, 1, 1)
        edge_rgb = edge.repeat(3, 1, 1)
        
        return {
            'nc': nc_rgb,
            'ce': ce_rgb,
            'edge': edge_rgb,
            'prompt': 'contrast-enhanced CT scan, medical imaging, high quality'
        }
class ControlNetTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        root = Path(args.root)
        self.exp_dir = Path(args.exp_dir)
        self.ckpt_dir = self.exp_dir / 'ckpt'
        self.samples_dir = self.exp_dir / 'samples'
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        model_id = "runwayml/stable-diffusion-v1-5"
        
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        self.controlnet = ControlNetModel.from_unet(self.unet).to(self.device)
        
        if args.use_lora:
            lora_config = LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                lora_dropout=0.1,
            )
            self.unet = get_peft_model(self.unet, lora_config)
        
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.scheduler.set_timesteps(args.num_inference_steps)
        
        self.structure_loss = StructureLoss().to(self.device)
        
        trainable_params = list(self.controlnet.parameters())
        if args.use_lora:
            trainable_params += list(filter(lambda p: p.requires_grad, self.unet.parameters()))
        
        self.optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
        
        self.scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-7
        )
        
        self.train_dataset = NCCEPairDataset(
            csv_path=root / args.pairs_csv,
            image_size=args.image_size,
            augment=True
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        
        if args.resume:
            self.load_checkpoint(args.resume)
    
    def load_checkpoint(self, ckpt_path):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            return
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.controlnet.load_state_dict(checkpoint['controlnet_state_dict'])
        if self.args.use_lora and checkpoint['unet_state_dict']:
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler_lr.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
    
    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        return text_embeddings
    
    @torch.no_grad()
    def encode_image(self, images):
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = latents / 0.18215
        images = self.vae.decode(latents).sample
        images = (images + 1) / 2
        return images.clamp(0, 1)
    
    def train_step(self, batch):
        nc = batch['nc'].to(self.device)
        ce = batch['ce'].to(self.device)
        edge = batch['edge'].to(self.device)
        prompts = batch['prompt']
        
        batch_size = nc.shape[0]
        
        text_embeddings = self.encode_prompt(prompts)
        
        nc_latents = self.encode_image(nc)
        ce_latents = self.encode_image(ce)
        
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, 
                                  (batch_size,), device=self.device).long()
        
        noise = torch.randn_like(ce_latents)
        noisy_latents = self.scheduler.add_noise(ce_latents, noise, timesteps)
        
        control_input = torch.cat([nc, edge], dim=1)
        
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=control_input,
            return_dict=False,
        )
        
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeddings,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        ).sample
        
        loss_mse = F.mse_loss(noise_pred, noise)
        
        with torch.no_grad():
            pred_latent = self.scheduler.step(noise_pred, timesteps[0], noisy_latents).pred_original_sample
            pred_image = self.decode_latents(pred_latent)
            nc_image = (nc + 1) / 2
        
        pred_hu = pred_image * 2 - 1
        nc_hu = nc_image * 2 - 1
        
        loss_structure = self.structure_loss(pred_hu, nc_hu)
        
        total_loss = loss_mse + self.args.structure_weight * loss_structure
        
        return total_loss, {
            'mse': loss_mse.item(),
            'structure': loss_structure.item(),
            'total': total_loss.item()
        }
    
    def train_epoch(self, epoch):
        self.controlnet.train()
        self.unet.train()
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        for batch in pbar:
            loss, loss_dict = self.train_step(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), max_norm=1.0)
            if self.args.use_lora:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.unet.parameters()), max_norm=1.0)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        
        avg_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_loss)
        self.scheduler_lr.step()
        
        return avg_loss
    @torch.no_grad()
    def generate_sample(self, nc, edge, prompt, num_steps=50):
        self.controlnet.eval()
        self.unet.eval()
        
        text_embeddings = self.encode_prompt([prompt])
        
        control_input = torch.cat([nc, edge], dim=1)
        
        latents = torch.randn((1, 4, self.args.image_size // 8, self.args.image_size // 8),
                             device=self.device)
        
        self.scheduler.set_timesteps(num_steps)
        for t in self.scheduler.timesteps:
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latents,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_input,
                return_dict=False,
            )
            
            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        image = self.decode_latents(latents)
        return image
    
    def save_samples(self, epoch):
        batch = next(iter(self.train_loader))
        nc = batch['nc'][:4].to(self.device)
        ce = batch['ce'][:4].to(self.device)
        edge = batch['edge'][:4].to(self.device)
        prompt = batch['prompt'][0]
        
        fake_ce = []
        for i in range(4):
            fake = self.generate_sample(nc[i:i+1], edge[i:i+1], prompt, num_steps=20)
            fake_ce.append(fake)
        fake_ce = torch.cat(fake_ce, dim=0)
        
        psnr_vals = []
        ssim_vals = []
        for i in range(4):
            fake_np = fake_ce[i].mean(0).cpu().numpy()
            real_np = ((ce[i].mean(0).cpu().numpy() + 1) / 2)
            psnr_vals.append(psnr_metric(real_np, fake_np, data_range=1.0))
            ssim_vals.append(ssim_metric(real_np, fake_np, data_range=1.0))
        
        avg_psnr = np.mean(psnr_vals)
        avg_ssim = np.mean(ssim_vals)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle(f'Epoch {epoch+1} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}', fontsize=14)
        
        for i in range(4):
            axes[i, 0].imshow(nc[i].mean(0).cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
            axes[i, 0].set_title('NC Input')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(edge[i].mean(0).cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
            axes[i, 1].set_title('Edge Map')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(fake_ce[i].mean(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Fake CE (PSNR: {psnr_vals[i]:.2f})')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow((ce[i].mean(0).cpu().numpy() + 1) / 2, cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title('Real CE')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.samples_dir / f'epoch_{epoch+1:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'controlnet_state_dict': self.controlnet.state_dict(),
            'unet_state_dict': self.unet.state_dict() if self.args.use_lora else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler_lr.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
        }
        
        torch.save(checkpoint, self.ckpt_dir / f'epoch_{epoch+1:03d}.pth')
        
        if is_best:
            torch.save(checkpoint, self.ckpt_dir / 'best.pth')
            self.controlnet.save_pretrained(self.ckpt_dir / 'best_controlnet')
    
    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            avg_loss = self.train_epoch(epoch)
            
            if (epoch + 1) % self.args.sample_interval == 0:
                self.save_samples(epoch)
            
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, is_best)


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--pairs-csv', type=str, default='Data/pseudo_pairs.csv')
    parser.add_argument('--exp-dir', type=str, required=True)
    parser.add_argument('--resume', type=str, default='')
    
    parser.add_argument('--use-lora', action='store_true', default=True)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--num-inference-steps', type=int, default=50)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num-workers', type=int, default=4)
    
    parser.add_argument('--structure-weight', type=float, default=0.1)
    
    parser.add_argument('--sample-interval', type=int, default=5)
    parser.add_argument('--save-interval', type=int, default=10)
    
    args = parser.parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    trainer = ControlNetTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()