# -*- coding: utf-8 -*-
"""
NC → CE Translation using Stable Diffusion + LoRA
학습된 LoRA 어댑터를 로드해서 NC CT를 CE 스타일로 변환합니다.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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
import lpips
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 이미지 저장


# ============================================================
# Dataset: NC CT 이미지 로드
# ============================================================
class NCCTDataset(Dataset):
    """NC CT 이미지를 로드하는 데이터셋"""
    
    def __init__(self, root, pairs_csv, split='test', image_size=512):
        self.root = Path(root)
        self.image_size = image_size
        
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
        
        self.samples = []
        for _, row in df.iterrows():
            # input_nc_norm 경로 사용
            nc_path = Path(row['input_nc_norm'])
            ce_path = Path(row['target_ce_norm'])  # Real CE 경로 추가
            
            if nc_path.exists() and ce_path.exists():
                self.samples.append({
                    'pid': row['id7'],
                    'nc_path': nc_path,
                    'ce_path': ce_path
                })
        
        print(f"[{split}] Loaded {len(self.samples)} NC CT volumes")
        
        # Transform - resize는 __getitem__에서 직접 처리
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # NC 볼륨 로드
        nc_img = sitk.ReadImage(str(sample['nc_path']))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        # CE 볼륨 로드 (Real CE)
        ce_img = sitk.ReadImage(str(sample['ce_path']))
        ce_arr = sitk.GetArrayFromImage(ce_img).astype(np.float32)
        
        # 중간 슬라이스 선택
        D_nc, H, W = nc_arr.shape
        D_ce = ce_arr.shape[0]
        
        mid_slice_nc = nc_arr[D_nc // 2]
        mid_slice_ce = ce_arr[D_ce // 2]
        
        # 데이터가 이미 [0, 1]로 정규화되어 있음
        # [-1, 1] 범위로만 변환
        mid_slice_nc = mid_slice_nc * 2.0 - 1.0  # [0, 1] → [-1, 1]
        mid_slice_ce = mid_slice_ce * 2.0 - 1.0
        
        # Tensor로 변환
        nc_tensor = torch.from_numpy(mid_slice_nc).unsqueeze(0)  # [1, H, W]
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
            'ce_real': ce_tensor,  # Real CE 추가
            'nc_raw': (mid_slice_nc + 1.0) / 2.0,  # [-1,1] → [0,1] for visualization
            'ce_real_raw': (mid_slice_ce + 1.0) / 2.0
        }


# ============================================================
# Inference Pipeline
# ============================================================
class SDLoRAInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        print("Loading Stable Diffusion components...")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float16
        ).to(self.device)
        self.vae.eval()
        
        # Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(self.device)
        self.text_encoder.eval()
        
        # UNet with LoRA
        print("Loading LoRA adapter...")
        base_unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch.float16
        )
        self.unet = PeftModel.from_pretrained(
            base_unet,
            self.checkpoint_path,
            is_trainable=False
        ).to(self.device)
        self.unet.eval()
        
        # Scheduler (DDIM for faster inference)
        self.scheduler = DDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.scheduler.set_timesteps(50)  # 50 steps (빠른 샘플링)
        
        # LPIPS model for perceptual similarity
        print("Loading LPIPS model...")
        self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        self.lpips_model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
    
    def encode_prompt(self, prompt):
        """텍스트를 임베딩으로 변환"""
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
    def translate_nc_to_ce(self, nc_image, strength=0.8):
        """
        NC 이미지를 CE 스타일로 변환
        
        Args:
            nc_image: [1, H, W] 텐서
            strength: 0-1, 높을수록 더 많이 변환 (0.8 권장)
        """
        # NC를 latent로 인코딩
        nc_rgb = nc_image.repeat(1, 3, 1, 1).to(self.device, dtype=torch.float16)
        
        init_latents = self.vae.encode(nc_rgb).latent_dist.sample()
        init_latents = init_latents * self.vae.config.scaling_factor
        
        # Prompt
        prompt = "contrast-enhanced CT scan, medical imaging, enhanced blood vessels"
        encoder_hidden_states = self.encode_prompt(prompt)
        
        # Noise 추가 (strength에 따라)
        noise = torch.randn_like(init_latents)
        timesteps = self.scheduler.timesteps
        start_step = int(len(timesteps) * (1 - strength))
        timestep = timesteps[start_step:start_step+1]
        
        latents = self.scheduler.add_noise(init_latents, noise, timestep)
        
        # Denoising loop
        for t in tqdm(timesteps[start_step:], desc="Translating", leave=False):
            latent_model_input = latents
            
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        
        return image
    
    def process_dataset(self, dataset, output_dir, strength=0.8):
        """데이터셋 전체 처리"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # 전체 지표 저장
        all_metrics = []
        
        for batch in tqdm(dataloader, desc="Processing volumes"):
            pid = batch['pid'][0]
            nc = batch['nc']  # [1, 1, H, W]
            nc_raw = batch['nc_raw'].numpy()[0]
            ce_real_raw = batch['ce_real_raw'].numpy()[0]
            
            # Translate
            ce_pred = self.translate_nc_to_ce(nc, strength=strength)
            
            # Post-process
            ce_pred_normalized = (ce_pred / 2 + 0.5).clamp(0, 1)
            ce_pred_np = ce_pred_normalized.cpu()[0, 0].numpy()  # 그레이스케일로
            
            # nc_raw를 ce_pred_np와 같은 크기로 resize (PSNR 계산용)
            from skimage.transform import resize as sk_resize
            if nc_raw.shape != ce_pred_np.shape:
                nc_raw_resized = sk_resize(nc_raw, ce_pred_np.shape, 
                                          order=1, preserve_range=True, anti_aliasing=True)
                ce_real_raw_resized = sk_resize(ce_real_raw, ce_pred_np.shape,
                                               order=1, preserve_range=True, anti_aliasing=True)
            else:
                nc_raw_resized = nc_raw
                ce_real_raw_resized = ce_real_raw
            
            # 지표 계산 1: Generated CE vs Real CE (기존)
            psnr_gen_real = psnr(ce_real_raw_resized, ce_pred_np, data_range=1.0)
            ssim_gen_real = ssim(ce_real_raw_resized, ce_pred_np, data_range=1.0)
            mae_gen_real = np.mean(np.abs(ce_real_raw_resized - ce_pred_np))
            
            # 지표 계산 2: NC vs Generated CE (개선도 측정)
            psnr_nc_gen = psnr(nc_raw_resized, ce_pred_np, data_range=1.0)
            ssim_nc_gen = ssim(nc_raw_resized, ce_pred_np, data_range=1.0)
            mae_nc_gen = np.mean(np.abs(nc_raw_resized - ce_pred_np))
            
            # LPIPS 계산 (Generated CE vs Real CE)
            ce_pred_rgb = ce_pred_normalized[:, 0:1, :, :].repeat(1, 3, 1, 1)  # [1, 3, H, W]
            
            ce_real_tensor = torch.from_numpy(ce_real_raw_resized).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            ce_real_rgb = ce_real_tensor.repeat(1, 3, 1, 1).to(self.device, dtype=torch.float32)  # [1, 3, H, W]
            
            ce_pred_rgb = (ce_pred_rgb * 2 - 1).to(dtype=torch.float32)  # [0, 1] → [-1, 1]
            ce_real_rgb = (ce_real_rgb * 2 - 1).to(dtype=torch.float32)
            
            with torch.no_grad():
                lpips_gen_real = self.lpips_model(ce_pred_rgb, ce_real_rgb).item()
            
            # LPIPS (NC vs Generated CE)
            nc_tensor = torch.from_numpy(nc_raw_resized).unsqueeze(0).unsqueeze(0)
            nc_rgb = nc_tensor.repeat(1, 3, 1, 1).to(self.device, dtype=torch.float32)
            nc_rgb = (nc_rgb * 2 - 1).to(dtype=torch.float32)
            
            with torch.no_grad():
                lpips_nc_gen = self.lpips_model(nc_rgb, ce_pred_rgb).item()
            
            metrics = {
                'pid': pid,
                # Generated vs Real CE
                'psnr_gen_real': psnr_gen_real,
                'ssim_gen_real': ssim_gen_real,
                'mae_gen_real': mae_gen_real,
                'lpips_gen_real': lpips_gen_real,
                # NC vs Generated CE (얼마나 변했는가)
                'psnr_nc_gen': psnr_nc_gen,
                'ssim_nc_gen': ssim_nc_gen,
                'mae_nc_gen': mae_nc_gen,
                'lpips_nc_gen': lpips_nc_gen
            }
            all_metrics.append(metrics)
            
            # Save comparison image with metrics (원본 크기 사용)
            self.save_comparison_with_metrics(
                nc_raw, ce_pred_np, ce_real_raw, metrics,
                output_dir / f'{pid}_comparison.png'
            )
            
            # Save CE prediction as numpy
            np.save(output_dir / f'{pid}_ce_pred.npy', ce_pred_np)
        
        # Save all metrics to CSV
        import pandas as pd
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(output_dir / 'metrics.csv', index=False)
        
        # Print summary
        print(f"\n{'='*80}")
        print("Metrics Summary:")
        print(f"\n[1] Generated CE vs Real CE (얼마나 Real CE와 비슷한가)")
        print(f"  PSNR:  {df_metrics['psnr_gen_real'].mean():.2f} ± {df_metrics['psnr_gen_real'].std():.2f} dB")
        print(f"  SSIM:  {df_metrics['ssim_gen_real'].mean():.4f} ± {df_metrics['ssim_gen_real'].std():.4f}")
        print(f"  MAE:   {df_metrics['mae_gen_real'].mean():.4f} ± {df_metrics['mae_gen_real'].std():.4f}")
        print(f"  LPIPS: {df_metrics['lpips_gen_real'].mean():.4f} ± {df_metrics['lpips_gen_real'].std():.4f}")
        
        print(f"\n[2] NC vs Generated CE (얼마나 변환됐는가)")
        print(f"  PSNR:  {df_metrics['psnr_nc_gen'].mean():.2f} ± {df_metrics['psnr_nc_gen'].std():.2f} dB")
        print(f"  SSIM:  {df_metrics['ssim_nc_gen'].mean():.4f} ± {df_metrics['ssim_nc_gen'].std():.4f}")
        print(f"  MAE:   {df_metrics['mae_nc_gen'].mean():.4f} ± {df_metrics['mae_nc_gen'].std():.4f}")
        print(f"  LPIPS: {df_metrics['lpips_nc_gen'].mean():.4f} ± {df_metrics['lpips_nc_gen'].std():.4f}")
        
        print(f"\n{'='*80}")
        print("Interpretation:")
        print("\n[Generated CE vs Real CE]")
        print("  - NC를 CE로 변환했을 때 Real CE와 얼마나 비슷한가")
        print("  - LPIPS 0.3-0.5: NC→CE translation의 일반적 범위")
        print("\n[NC vs Generated CE]")
        print("  - 변환으로 얼마나 달라졌는가 (변화량)")
        print("  - LPIPS > 0.2: 충분한 변환이 일어남")
        print("  - MAE > 0.05: 시각적으로 의미있는 변화")
        print(f"{'='*80}\n")
        
        print(f"✓ Results saved to {output_dir}")
    
    def save_comparison_with_metrics(self, nc, ce_pred, ce_real, metrics, save_path):
        """NC, Generated CE, Real CE를 나란히 저장하고 지표 표시"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # NC
        axes[0].imshow(nc, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('NC (Input)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Generated CE
        axes[1].imshow(ce_pred, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Generated CE', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Real CE
        axes[2].imshow(ce_real, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Real CE (Ground Truth)', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        # 지표 텍스트 (2줄)
        metrics_text_line1 = (
            f"Gen vs Real: PSNR {metrics['psnr_gen_real']:.2f} dB | "
            f"SSIM {metrics['ssim_gen_real']:.4f} | "
            f"LPIPS {metrics['lpips_gen_real']:.4f}"
        )
        metrics_text_line2 = (
            f"NC vs Gen: PSNR {metrics['psnr_nc_gen']:.2f} dB | "
            f"SSIM {metrics['ssim_nc_gen']:.4f} | "
            f"LPIPS {metrics['lpips_nc_gen']:.4f}"
        )
        
        fig.text(0.5, 0.05, metrics_text_line1, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        fig.text(0.5, 0.01, metrics_text_line2, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0.10, 1, 1])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='NC → CE Translation using SD + LoRA')
    
    # Paths
    parser.add_argument('--root', default=r'E:\LD-CT SR')
    parser.add_argument('--pairs-csv', default='Data/pairs.csv')
    parser.add_argument('--checkpoint', required=True, help='Path to LoRA checkpoint')
    parser.add_argument('--output-dir', default='Outputs/translations/sd_lora')
    
    # Inference
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--strength', type=float, default=0.8, help='Translation strength (0-1)')
    parser.add_argument('--image-size', type=int, default=512)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print("NC → CE Translation with Stable Diffusion + LoRA")
    print(f"{'='*80}\n")
    
    # Load model
    pipeline = SDLoRAInference(args.checkpoint)
    
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
        strength=args.strength
    )


if __name__ == '__main__':
    main()