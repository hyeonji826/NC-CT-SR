#!/usr/bin/env python3
"""
14_inference_2stage_pipeline.py
2-Stage Pipeline: NAFNet Denoising → ControlNet Style Transfer
NC-CT → Denoised NC → Fake CE-CT
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    ControlNetModel
)
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms


# ============================================================
# NAFNet 모델 정의 (Stage 1)
# ============================================================
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        
        return y + x * self.gamma


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=32, middle_blk_num=12, 
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, 1, 1, bias=True)
        
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2
        
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(chan, chan // 2, 3, 1, 1, bias=True)
            ))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        
        x = self.intro(inp)
        
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        
        x = self.middle_blks(x)
        
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        
        x = self.ending(x)
        x = x + inp
        
        return x[:, :, :H, :W]
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ============================================================
# 2-Stage Pipeline
# ============================================================
class TwoStagePipeline:
    def __init__(self, nafnet_checkpoint, controlnet_checkpoint, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*80}")
        print("Loading 2-Stage Pipeline")
        print(f"{'='*80}\n")
        
        # Stage 1: NAFNet
        print("Stage 1: Loading NAFNet (Denoising)...")
        self.nafnet = self.load_nafnet(nafnet_checkpoint)
        print("  ✓ NAFNet loaded!\n")
        
        # Stage 2: ControlNet + SD
        print("Stage 2: Loading ControlNet + Stable Diffusion (Style Transfer)...")
        self.load_controlnet_sd(controlnet_checkpoint)
        print("  ✓ ControlNet + SD loaded!\n")
        
        print(f"{'='*80}\n")
    
    def load_nafnet(self, checkpoint_path):
        """NAFNet 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        config = checkpoint.get('config', {
            'width': 32,
            'middle_blk_num': 12,
            'enc_blk_nums': [2, 2, 4, 8],
            'dec_blk_nums': [2, 2, 2, 2]
        })
        
        model = NAFNet(
            img_channel=3,
            width=config['width'],
            middle_blk_num=config['middle_blk_num'],
            enc_blk_nums=config['enc_blk_nums'],
            dec_blk_nums=config['dec_blk_nums']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        if 'best_psnr' in checkpoint:
            print(f"  Best PSNR: {checkpoint['best_psnr']:.2f} dB")
        
        return model
    
    def load_controlnet_sd(self, checkpoint_dir):
        """ControlNet + SD 로드"""
        checkpoint_dir = Path(checkpoint_dir)
        
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        
        # Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float32
        ).to(self.device)
        self.text_encoder.eval()
        
        # ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            checkpoint_dir / 'controlnet',
            torch_dtype=torch.float32
        ).to(self.device)
        self.controlnet.eval()
        
        # UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            checkpoint_dir / 'unet_lora',
            torch_dtype=torch.float32
        ).to(self.device)
        self.unet.eval()
        
        # Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.scheduler.set_timesteps(50)
    
    @torch.no_grad()
    def denoise_slice(self, noisy_slice):
        """Stage 1: NAFNet Denoising"""
        # [H, W] → [1, 3, H, W]
        slice_tensor = torch.from_numpy(noisy_slice).unsqueeze(0).unsqueeze(0)
        slice_tensor = slice_tensor.repeat(1, 3, 1, 1).to(self.device, dtype=torch.float32)
        
        denoised = self.nafnet(slice_tensor)
        
        # [1, 3, H, W] → [H, W]
        return denoised[0, 0].cpu().numpy()
    
    @torch.no_grad()
    def style_transfer(self, denoised_slice, image_size=512):
        """Stage 2: ControlNet Style Transfer"""
        # Prepare control image
        control_np = denoised_slice * 2.0 - 1.0  # [0,1] → [-1,1]
        control_tensor = torch.from_numpy(control_np).unsqueeze(0).unsqueeze(0)
        control_tensor = control_tensor.repeat(1, 3, 1, 1).to(self.device, dtype=torch.float32)
        
        # Resize
        control_tensor = transforms.functional.resize(
            control_tensor, (image_size, image_size), antialias=True
        )
        
        # Encode control
        control_latents = self.vae.encode(control_tensor).latent_dist.sample()
        control_latents = control_latents * self.vae.config.scaling_factor
        
        # Text prompt
        prompt = "high quality contrast-enhanced CT scan with clear tissue boundaries and enhanced blood vessels"
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        encoder_hidden_states = self.text_encoder(
            text_inputs.input_ids.to(self.device)
        )[0]
        
        # Random latent
        latents = torch.randn(
            1, 4, image_size // 8, image_size // 8,
            device=self.device, dtype=torch.float32
        )
        
        # Denoising loop
        for t in self.scheduler.timesteps:
            # ControlNet
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latents, t, encoder_hidden_states,
                controlnet_cond=control_latents, return_dict=False
            )
            
            # UNet
            noise_pred = self.unet(
                latents, t, encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample
            ).sample
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        latents = 1 / self.vae.config.scaling_factor * latents
        generated = self.vae.decode(latents).sample
        
        # [1, 3, H, W] → [H, W]
        generated_np = generated[0, 0].cpu().numpy()
        generated_np = (generated_np / 2 + 0.5).clip(0, 1)
        
        # Resize back to original
        if generated_np.shape != denoised_slice.shape:
            from skimage.transform import resize
            generated_np = resize(
                generated_np, denoised_slice.shape,
                order=1, preserve_range=True, anti_aliasing=True
            )
        
        return generated_np
    
    def process_volume(self, nc_path, output_dir, patient_id, save_intermediate=True):
        """전체 볼륨 처리"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load NC volume
        nc_img = sitk.ReadImage(str(nc_path))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        D, H, W = nc_arr.shape
        
        denoised_volume = np.zeros_like(nc_arr)
        fake_ce_volume = np.zeros_like(nc_arr)
        
        print(f"\nProcessing: {patient_id}")
        print(f"  Volume shape: {nc_arr.shape}")
        
        for slice_idx in tqdm(range(D), desc=f"  [{patient_id}]"):
            original_slice = nc_arr[slice_idx]
            
            # Stage 1: Denoise
            denoised_slice = self.denoise_slice(original_slice)
            denoised_volume[slice_idx] = denoised_slice
            
            # Stage 2: Style Transfer (middle slice for visualization)
            if slice_idx == D // 2:
                fake_ce_slice = self.style_transfer(denoised_slice)
                fake_ce_volume[slice_idx] = fake_ce_slice
                
                # Save comparison
                self.save_comparison(
                    original_slice, denoised_slice, fake_ce_slice,
                    output_dir / 'samples' / f'{patient_id}_slice_{slice_idx:03d}.png',
                    patient_id, slice_idx
                )
        
        # Save volumes
        if save_intermediate:
            # Denoised volume
            denoised_sitk = sitk.GetImageFromArray(denoised_volume)
            denoised_sitk.CopyInformation(nc_img)
            sitk.WriteImage(
                denoised_sitk,
                str(output_dir / 'denoised' / f'{patient_id}_denoised.nii.gz')
            )
        
        # Fake CE volume (only middle slice for now)
        # For full volume, process all slices in Stage 2
        
        print(f"  ✓ Processed: {patient_id}")
    
    def save_comparison(self, original, denoised, fake_ce, output_path, patient_id, slice_idx):
        """3단계 비교 이미지 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original NC (Noisy)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(denoised, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Denoised NC (Clean)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(fake_ce, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Fake CE-CT (Style Transfer)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Patient: {patient_id} | Slice: {slice_idx}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='2-Stage Pipeline: NAFNet Denoising → ControlNet Style Transfer'
    )
    
    # Model checkpoints
    parser.add_argument('--nafnet-checkpoint', type=str, required=True,
                       help='Path to NAFNet checkpoint (Stage 1)')
    parser.add_argument('--controlnet-checkpoint', type=str, required=True,
                       help='Path to ControlNet checkpoint directory (Stage 2)')
    
    # Data
    parser.add_argument('--root-dir', type=str, default=r'E:\LD-CT SR',
                       help='Root directory')
    parser.add_argument('--pairs-csv', type=str, default='Data/pairs.csv',
                       help='Path to pairs.csv')
    
    # Output
    parser.add_argument('--output-dir', type=str, 
                       default='Outputs/inference_2stage',
                       help='Output directory')
    
    # Options
    parser.add_argument('--num-test', type=int, default=5,
                       help='Number of patients to test')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size for ControlNet')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    
    args = parser.parse_args()
    
    # Load pipeline
    pipeline = TwoStagePipeline(
        args.nafnet_checkpoint,
        args.controlnet_checkpoint,
        device=args.device
    )
    
    # Load pairs
    root = Path(args.root_dir)
    pairs_df = pd.read_csv(root / args.pairs_csv)
    
    # Create output directories
    output_dir = root / args.output_dir
    (output_dir / 'denoised').mkdir(parents=True, exist_ok=True)
    (output_dir / 'samples').mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("2-Stage Inference Pipeline")
    print(f"  Stage 1: NAFNet Denoising")
    print(f"  Stage 2: ControlNet Style Transfer")
    print(f"{'='*80}\n")
    
    # Process patients
    num_test = min(args.num_test, len(pairs_df))
    
    for idx in range(num_test):
        row = pairs_df.iloc[idx]
        patient_id = row['id7']
        nc_path = Path(row['input_nc_norm'])
        
        if not nc_path.exists():
            print(f"⚠️  Skipping {patient_id}: NC file not found")
            continue
        
        pipeline.process_volume(
            nc_path, output_dir, patient_id,
            save_intermediate=True
        )
    
    print(f"\n{'='*80}")
    print("Inference completed!")
    print(f"  Denoised volumes: {output_dir / 'denoised'}")
    print(f"  Sample comparisons: {output_dir / 'samples'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()