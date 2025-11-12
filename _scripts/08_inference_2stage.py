# -*- coding: utf-8 -*-
"""
2-Stage NC Enhancement Pipeline (Updated)
Stage 1: NC → Enhanced NC (NAFNet - 구조 완벽 보존)
Stage 2: Enhanced NC → Pseudo CE (SD LoRA - 약한 조영 효과)
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch.utils._pytree.*')

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler
)
from peft import PeftModel
from transformers import CLIPTextModel, CLIPTokenizer
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize as sk_resize
import lpips
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# ============================================================
# NAFNet (from 09_train_nafnet.py)
# ============================================================
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )
        
        self.sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
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
        
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=1, width=32, middle_blk_num=12, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        
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
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
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
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ============================================================
# Dataset: NC CT only
# ============================================================
class NCCTDataset(Dataset):
    """NC CT 이미지만 로드"""
    
    def __init__(self, root, pairs_csv, split='test', image_size=512):
        self.root = Path(root)
        self.image_size = image_size
        
        df = pd.read_csv(self.root / pairs_csv)
        
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
        
        self.samples = []
        for _, row in df.iterrows():
            nc_path = Path(row['input_nc_norm'])
            ce_path = Path(row['target_ce_norm'])
            
            if nc_path.exists() and ce_path.exists():
                self.samples.append({
                    'pid': row['id7'],
                    'nc_path': nc_path,
                    'ce_path': ce_path
                })
        
        print(f"[{split}] Loaded {len(self.samples)} NC CT volumes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # NC 볼륨 로드
        nc_img = sitk.ReadImage(str(sample['nc_path']))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        # CE 볼륨 로드 (비교용)
        ce_img = sitk.ReadImage(str(sample['ce_path']))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        
        # 중간 슬라이스
        D_nc, H, W = nc_arr.shape
        D_ce = ce_arr.shape[0]
        
        mid_slice_nc = nc_arr[D_nc // 2]
        mid_slice_ce = ce_arr[D_ce // 2]
        
        # [-1, 1] 변환
        mid_slice_nc = mid_slice_nc * 2.0 - 1.0
        mid_slice_ce = mid_slice_ce * 2.0 - 1.0
        
        # Tensor
        nc_tensor = torch.from_numpy(mid_slice_nc).unsqueeze(0)
        ce_tensor = torch.from_numpy(mid_slice_ce).unsqueeze(0)
        
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
        
        return {
            'pid': sample['pid'],
            'nc': nc_tensor,
            'ce_real': ce_tensor,
            'nc_raw': (mid_slice_nc + 1.0) / 2.0,
            'ce_real_raw': (mid_slice_ce + 1.0) / 2.0
        }


# ============================================================
# 2-Stage Pipeline (NAFNet + SD LoRA)
# ============================================================
class TwoStagePipeline:
    def __init__(self, nafnet_checkpoint, style_checkpoint, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Stage 1: NAFNet
        print("Loading Stage 1 (NAFNet) model...")
        self.nafnet = NAFNet(
            img_channel=1,
            width=32,
            middle_blk_num=12,
            enc_blk_nums=[2, 2, 4, 8],
            dec_blk_nums=[2, 2, 2, 2]
        ).to(self.device)
        
        checkpoint = torch.load(nafnet_checkpoint, map_location=self.device)
        self.nafnet.load_state_dict(checkpoint['model'])
        self.nafnet.eval()
        
        # Stage 2: SD LoRA
        print("Loading Stage 2 (SD LoRA) model...")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float32
        ).to(self.device)
        self.text_encoder.eval()
        
        base_unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float32
        )
        self.unet_style = PeftModel.from_pretrained(
            base_unet,
            style_checkpoint,
            is_trainable=False
        ).to(self.device)
        self.unet_style.eval()
        
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.scheduler.set_timesteps(50)
        
        # LPIPS
        print("Loading LPIPS model...")
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_model.eval()
        
        print(f"✓ 2-Stage Pipeline loaded")
    
    def encode_prompt(self, prompt):
        text_inputs = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(self.device)
        
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        return prompt_embeds
    
    @torch.no_grad()
    def enhance_nc(self, nc_image):
        """Stage 1: NC → Enhanced NC (NAFNet - 구조 보존)"""
        # nc_image: [1, 1, H, W], [0, 1] range
        enhanced = self.nafnet(nc_image)
        return enhanced
    
    @torch.no_grad()
    def apply_ce_style(self, enhanced_nc, strength=0.3):
        """Stage 2: Enhanced NC → Pseudo CE (약한 조영 효과)"""
        # enhanced_nc: [1, 1, H, W], [0, 1] range
        # SD는 [-1, 1] 범위 필요
        enhanced_nc_scaled = enhanced_nc * 2.0 - 1.0
        enhanced_rgb = enhanced_nc_scaled.repeat(1, 3, 1, 1).to(self.device, dtype=torch.float32)
        
        latents = self.vae.encode(enhanced_rgb).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        
        prompt = "contrast-enhanced CT scan, medical imaging, enhanced blood vessels"
        encoder_hidden_states = self.encode_prompt(prompt)
        
        # Add noise (약하게)
        noise = torch.randn_like(latents)
        timesteps = self.scheduler.timesteps
        start_step = int(len(timesteps) * (1 - strength))
        timestep = timesteps[start_step:start_step+1]
        
        latents = self.scheduler.add_noise(latents, noise, timestep)
        
        # Denoising
        for t in timesteps[start_step:]:
            latent_model_input = latents
            
            noise_pred = self.unet_style(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        
        # [-1, 1] → [0, 1]
        image = (image + 1.0) / 2.0
        
        return image
    
    def process_dataset(self, dataset, output_dir, style_strength=0.3):
        """데이터셋 전체 처리"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        all_metrics = []
        
        for batch in tqdm(dataloader, desc="Processing 2-stage pipeline"):
            pid = batch['pid'][0]
            nc = batch['nc']  # [1, 1, H, W], [0, 1]
            nc_raw = batch['nc_raw'].numpy()[0]
            ce_real_raw = batch['ce_real_raw'].numpy()[0]
            
            # Stage 1: NC → Enhanced NC (NAFNet)
            enhanced_nc = self.enhance_nc(nc.to(self.device))
            enhanced_nc_np = enhanced_nc.cpu()[0, 0].numpy()
            
            # Stage 2: Enhanced NC → Pseudo CE
            pseudo_ce = self.apply_ce_style(enhanced_nc, strength=style_strength)
            pseudo_ce_np = pseudo_ce.cpu()[0, 0].numpy()
            
            # Resize for metrics
            if nc_raw.shape != enhanced_nc_np.shape:
                nc_raw_resized = sk_resize(nc_raw, enhanced_nc_np.shape, 
                                          order=1, preserve_range=True, anti_aliasing=True)
                ce_real_raw_resized = sk_resize(ce_real_raw, enhanced_nc_np.shape,
                                               order=1, preserve_range=True, anti_aliasing=True)
            else:
                nc_raw_resized = nc_raw
                ce_real_raw_resized = ce_real_raw
            
            # 지표 계산
            # 1) Original NC vs Enhanced NC
            psnr_nc_enh = psnr(nc_raw_resized, enhanced_nc_np, data_range=1.0)
            ssim_nc_enh = ssim(nc_raw_resized, enhanced_nc_np, data_range=1.0)
            mae_nc_enh = np.mean(np.abs(nc_raw_resized - enhanced_nc_np))
            
            # 2) Enhanced NC vs Pseudo CE
            psnr_enh_pce = psnr(enhanced_nc_np, pseudo_ce_np, data_range=1.0)
            ssim_enh_pce = ssim(enhanced_nc_np, pseudo_ce_np, data_range=1.0)
            mae_enh_pce = np.mean(np.abs(enhanced_nc_np - pseudo_ce_np))
            
            # 3) Pseudo CE vs Real CE
            psnr_pce_real = psnr(ce_real_raw_resized, pseudo_ce_np, data_range=1.0)
            ssim_pce_real = ssim(ce_real_raw_resized, pseudo_ce_np, data_range=1.0)
            mae_pce_real = np.mean(np.abs(ce_real_raw_resized - pseudo_ce_np))
            
            # 4) Original NC vs Real CE (baseline)
            psnr_nc_real = psnr(nc_raw_resized, ce_real_raw_resized, data_range=1.0)
            ssim_nc_real = ssim(nc_raw_resized, ce_real_raw_resized, data_range=1.0)
            
            # LPIPS
            def compute_lpips(img1, img2):
                t1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                t2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
                t1 = (t1 * 2 - 1).to(self.device, dtype=torch.float32)
                t2 = (t2 * 2 - 1).to(self.device, dtype=torch.float32)
                return self.lpips_model(t1, t2).item()
            
            lpips_nc_enh = compute_lpips(nc_raw_resized, enhanced_nc_np)
            lpips_enh_pce = compute_lpips(enhanced_nc_np, pseudo_ce_np)
            lpips_pce_real = compute_lpips(pseudo_ce_np, ce_real_raw_resized)
            lpips_nc_real = compute_lpips(nc_raw_resized, ce_real_raw_resized)
            
            metrics = {
                'pid': pid,
                # Stage 1: NC → Enhanced
                'psnr_nc_enh': psnr_nc_enh,
                'ssim_nc_enh': ssim_nc_enh,
                'mae_nc_enh': mae_nc_enh,
                'lpips_nc_enh': lpips_nc_enh,
                # Stage 2: Enhanced → Pseudo CE
                'psnr_enh_pce': psnr_enh_pce,
                'ssim_enh_pce': ssim_enh_pce,
                'mae_enh_pce': mae_enh_pce,
                'lpips_enh_pce': lpips_enh_pce,
                # Final: Pseudo CE vs Real CE
                'psnr_pce_real': psnr_pce_real,
                'ssim_pce_real': ssim_pce_real,
                'mae_pce_real': mae_pce_real,
                'lpips_pce_real': lpips_pce_real,
                # Baseline: NC vs Real CE
                'psnr_nc_real': psnr_nc_real,
                'ssim_nc_real': ssim_nc_real,
                'lpips_nc_real': lpips_nc_real
            }
            all_metrics.append(metrics)
            
            # Save visualization
            self.save_comparison(
                nc_raw, enhanced_nc_np, pseudo_ce_np, ce_real_raw, metrics,
                output_dir / f'{pid}_2stage.png'
            )
            
            # Save outputs
            np.save(output_dir / f'{pid}_enhanced_nc.npy', enhanced_nc_np)
            np.save(output_dir / f'{pid}_pseudo_ce.npy', pseudo_ce_np)
        
        # Save metrics CSV
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(output_dir / 'metrics_2stage.csv', index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("2-Stage Pipeline Metrics Summary:")
        print(f"\n[Stage 1] NC → Enhanced NC (화질 개선)")
        print(f"  PSNR:  {df_metrics['psnr_nc_enh'].mean():.2f} ± {df_metrics['psnr_nc_enh'].std():.2f} dB")
        print(f"  SSIM:  {df_metrics['ssim_nc_enh'].mean():.4f} ± {df_metrics['ssim_nc_enh'].std():.4f}")
        print(f"  LPIPS: {df_metrics['lpips_nc_enh'].mean():.4f} ± {df_metrics['lpips_nc_enh'].std():.4f}")
        
        print(f"\n[Stage 2] Enhanced NC → Pseudo CE (조영 효과)")
        print(f"  PSNR:  {df_metrics['psnr_enh_pce'].mean():.2f} ± {df_metrics['psnr_enh_pce'].std():.2f} dB")
        print(f"  SSIM:  {df_metrics['ssim_enh_pce'].mean():.4f} ± {df_metrics['ssim_enh_pce'].std():.4f}")
        print(f"  LPIPS: {df_metrics['lpips_enh_pce'].mean():.4f} ± {df_metrics['lpips_enh_pce'].std():.4f}")
        
        print(f"\n[Final] Pseudo CE vs Real CE")
        print(f"  PSNR:  {df_metrics['psnr_pce_real'].mean():.2f} ± {df_metrics['psnr_pce_real'].std():.2f} dB")
        print(f"  SSIM:  {df_metrics['ssim_pce_real'].mean():.4f} ± {df_metrics['ssim_pce_real'].std():.4f}")
        print(f"  LPIPS: {df_metrics['lpips_pce_real'].mean():.4f} ± {df_metrics['lpips_pce_real'].std():.4f}")
        
        print(f"\n[Baseline] Original NC vs Real CE")
        print(f"  PSNR:  {df_metrics['psnr_nc_real'].mean():.2f} ± {df_metrics['psnr_nc_real'].std():.2f} dB")
        print(f"  SSIM:  {df_metrics['ssim_nc_real'].mean():.4f} ± {df_metrics['ssim_nc_real'].std():.4f}")
        print(f"  LPIPS: {df_metrics['lpips_nc_real'].mean():.4f} ± {df_metrics['lpips_nc_real'].std():.4f}")
        print(f"{'='*80}\n")
        
        print(f"✓ Results saved to {output_dir}")
    
    def save_comparison(self, nc, enhanced_nc, pseudo_ce, real_ce, metrics, save_path):
        """4개 이미지 비교 + 지표"""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original NC
        axes[0].imshow(nc, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Original NC', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Enhanced NC
        axes[1].imshow(enhanced_nc, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Enhanced NC\n(Stage 1)', fontsize=12, fontweight='bold', color='green')
        axes[1].axis('off')
        
        # Pseudo CE
        axes[2].imshow(pseudo_ce, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Pseudo CE\n(Stage 2)', fontsize=12, fontweight='bold', color='blue')
        axes[2].axis('off')
        
        # Real CE
        axes[3].imshow(real_ce, cmap='gray', vmin=0, vmax=1)
        axes[3].set_title('Real CE\n(Ground Truth)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        # Metrics text (3줄)
        line1 = f"Stage 1 (NC→Enh): PSNR {metrics['psnr_nc_enh']:.2f} | SSIM {metrics['ssim_nc_enh']:.4f} | LPIPS {metrics['lpips_nc_enh']:.4f}"
        line2 = f"Stage 2 (Enh→PCE): PSNR {metrics['psnr_enh_pce']:.2f} | SSIM {metrics['ssim_enh_pce']:.4f} | LPIPS {metrics['lpips_enh_pce']:.4f}"
        line3 = f"Final (PCE vs Real): PSNR {metrics['psnr_pce_real']:.2f} | SSIM {metrics['ssim_pce_real']:.4f} | LPIPS {metrics['lpips_pce_real']:.4f}"
        
        fig.text(0.5, 0.08, line1, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        fig.text(0.5, 0.04, line2, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        fig.text(0.5, 0.00, line3, ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='2-Stage NC Enhancement Pipeline (NAFNet + SD LoRA)')
    
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--nafnet-checkpoint', required=True, 
                       help='Path to NAFNet checkpoint (best.pth)')
    parser.add_argument('--style-checkpoint', required=True,
                       help='Path to SD LoRA CE style checkpoint')
    parser.add_argument('--output-dir', default='Outputs/translations/2stage_nafnet')
    
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--style-strength', type=float, default=0.3,
                       help='Stage 2 CE style strength (0-1, lower = less CE effect)')
    parser.add_argument('--image-size', type=int, default=512)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("2-Stage NC Enhancement Pipeline (NAFNet + SD LoRA)")
    print(f"  Stage 1: NAFNet (구조 보존 화질 개선)")
    print(f"  Stage 2: SD LoRA (CE style, strength={args.style_strength})")
    print(f"{'='*80}\n")
    
    # Load pipeline
    pipeline = TwoStagePipeline(
        args.nafnet_checkpoint,
        args.style_checkpoint
    )
    
    # Load dataset
    dataset = NCCTDataset(
        args.root,
        args.pairs_csv,
        split=args.split,
        image_size=args.image_size
    )
    
    # Process
    pipeline.process_dataset(
        dataset,
        args.output_dir,
        style_strength=args.style_strength
    )


if __name__ == '__main__':
    main()