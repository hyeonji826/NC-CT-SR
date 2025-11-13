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
import matplotlib.pyplot as plt

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer


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
        y, var, weight = ctx.saved_tensors
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
    def __init__(self, img_channel=1, width=32, middle_blk_num=12, enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
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
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)
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
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


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


class TestDataset(Dataset):
    def __init__(self, csv_path, image_size=512):
        self.pairs = pd.read_csv(csv_path)
        self.image_size = image_size
        
    def __len__(self):
        return len(self.pairs)
    
    def load_slice(self, nifti_path, slice_idx):
        img = sitk.ReadImage(str(nifti_path))
        arr = sitk.GetArrayFromImage(img)[slice_idx]
        arr = np.clip((arr + 1000) / 1000, -1, 1)
        return arr
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        nc = self.load_slice(row['nc_path'], row['nc_slice_idx'])
        nc = cv2.resize(nc, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        nc_tensor = torch.from_numpy(nc).float().unsqueeze(0)
        
        return {
            'nc': nc_tensor,
            'nc_path': row['nc_path'],
            'slice_idx': row['nc_slice_idx']
        }
class TwoStagePipeline:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.nafnet = NAFNet(img_channel=1, width=32).to(self.device)
        checkpoint = torch.load(args.nafnet_checkpoint, map_location=self.device)
        self.nafnet.load_state_dict(checkpoint['model_state_dict'])
        self.nafnet.eval()
        
        model_id = "runwayml/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        
        self.controlnet = ControlNetModel.from_pretrained(args.controlnet_checkpoint).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        self.vae.eval()
        self.text_encoder.eval()
        self.unet.eval()
        self.controlnet.eval()
        
    @torch.no_grad()
    def stage1_denoise(self, nc):
        nc_norm = (nc + 1) / 2
        denoised = self.nafnet(nc_norm)
        denoised = denoised * 2 - 1
        return denoised
    
    @torch.no_grad()
    def stage2_style_transfer(self, denoised_nc, num_steps=50):
        prompt = "contrast-enhanced CT scan, medical imaging, high quality"
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        edge = extract_edges(denoised_nc)
        
        nc_rgb = denoised_nc.repeat(1, 3, 1, 1)
        edge_rgb = edge.repeat(1, 3, 1, 1)
        
        control_input = torch.cat([nc_rgb, edge_rgb], dim=1)
        
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
        
        latents = latents / 0.18215
        image = self.vae.decode(latents).sample
        image = (image + 1) / 2
        
        return image.clamp(0, 1)
    
    def run(self, test_loader, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        samples_dir = output_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)
        
        for idx, batch in enumerate(tqdm(test_loader, desc="Inference")):
            if idx >= self.args.num_test:
                break
            
            nc = batch['nc'].to(self.device)
            
            denoised_nc = self.stage1_denoise(nc)
            
            fake_ce = self.stage2_style_transfer(denoised_nc, num_steps=self.args.num_inference_steps)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow((nc[0, 0].cpu().numpy() + 1) / 2, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('Original NC (Noisy)', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow((denoised_nc[0, 0].cpu().numpy() + 1) / 2, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Denoised NC (Stage-1)', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(fake_ce[0].mean(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Fake CE (Stage-2)', fontsize=12)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(samples_dir / f'sample_{idx:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--nafnet-checkpoint', type=str, required=True)
    parser.add_argument('--controlnet-checkpoint', type=str, required=True)
    parser.add_argument('--root-dir', type=str, required=True)
    parser.add_argument('--pairs-csv', type=str, default='Data/pseudo_pairs.csv')
    parser.add_argument('--output-dir', type=str, required=True)
    
    parser.add_argument('--num-test', type=int, default=10)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--num-inference-steps', type=int, default=50)
    
    args = parser.parse_args()
    
    test_dataset = TestDataset(
        csv_path=Path(args.root_dir) / args.pairs_csv,
        image_size=args.image_size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    pipeline = TwoStagePipeline(args)
    pipeline.run(test_loader, args.output_dir)


if __name__ == '__main__':
    main()