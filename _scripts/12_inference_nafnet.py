#!/usr/bin/env python3
"""
NAFNet Denoising Inference Script
학습된 Noise2Noise 모델로 NC-CT denoising 수행
Stage 1 결과: Clean NC-CT 생성
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import yaml
import pandas as pd


class NAFBlock(nn.Module):
    """NAFNet의 기본 블록"""
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
    """Simple Gate activation"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNorm2d(nn.Module):
    """2D Layer Normalization"""
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
    """NAFNet 모델"""
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
        x = torch.nn.functional.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class InferenceDataset(Dataset):
    """Inference용 데이터셋 - 3D CT volumes 처리"""
    def __init__(self, csv_path, root_dir):
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir)
        self.pairs_df = pd.read_csv(csv_path)
        
        # Get unique NC volumes
        self.nc_volumes = self.pairs_df['input_nc_norm'].unique()
        print(f"Found {len(self.nc_volumes)} NC volumes to process")
    
    def __len__(self):
        return len(self.nc_volumes)
    
    def __getitem__(self, idx):
        nc_path = Path(self.nc_volumes[idx])
        
        # Load full volume
        import SimpleITK as sitk
        nc_img = sitk.ReadImage(str(nc_path))
        nc_arr = sitk.GetArrayFromImage(nc_img).astype(np.float32)
        
        # Get patient ID from path
        patient_id = nc_path.stem.split('_')[0]
        
        return {
            'volume': nc_arr,
            'path': str(nc_path),
            'patient_id': patient_id,
            'sitk_image': nc_img  # Keep original for saving with metadata
        }


def load_model(checkpoint_path, device):
    """모델 로드"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 설정 로드
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # 기본 설정
        config = {
            'width': 32,
            'middle_blk_num': 12,
            'enc_blk_nums': [2, 2, 4, 8],
            'dec_blk_nums': [2, 2, 2, 2]
        }
    
    # 모델 생성 및 가중치 로드
    model = NAFNet(
        img_channel=3,
        width=config['width'],
        middle_blk_num=config['middle_blk_num'],
        enc_blk_nums=config['enc_blk_nums'],
        dec_blk_nums=config['dec_blk_nums']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    if 'best_psnr' in checkpoint:
        print(f"Best PSNR: {checkpoint['best_psnr']:.2f} dB")
    
    return model


def inference(model, dataloader, output_dir, device):
    """Inference 수행 - 3D volumes을 slice-by-slice로 처리"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting denoising inference...")
    print(f"Output directory: {output_dir}")
    
    import SimpleITK as sitk
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing volumes"):
            # Batch size는 1로 가정 (volumes은 크기가 다를 수 있음)
            volume = batch['volume'][0].numpy()  # [D, H, W]
            patient_id = batch['patient_id'][0]
            original_path = batch['path'][0]
            sitk_image = batch['sitk_image']
            
            D, H, W = volume.shape
            denoised_volume = np.zeros_like(volume)
            
            # Process each slice
            for slice_idx in tqdm(range(D), desc=f"  [{patient_id}]", leave=False):
                # Prepare slice
                slice_2d = volume[slice_idx]  # [H, W]
                
                # Convert to tensor [1, 3, H, W] (batch=1, RGB channels)
                slice_tensor = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                slice_tensor = slice_tensor.repeat(1, 3, 1, 1)  # [1, 3, H, W]
                slice_tensor = slice_tensor.to(device)
                
                # Denoise
                denoised_slice = model(slice_tensor)
                
                # Extract first channel and convert back to numpy
                denoised_slice = denoised_slice[0, 0].cpu().numpy()  # [H, W]
                denoised_volume[slice_idx] = denoised_slice
            
            # Save denoised volume as NIfTI with original metadata
            output_filename = f"{patient_id}_denoised.nii.gz"
            output_path = output_dir / output_filename
            
            # Create SimpleITK image with denoised data but original metadata
            denoised_sitk = sitk.GetImageFromArray(denoised_volume.astype(np.float32))
            
            # Copy metadata from original
            if len(sitk_image) > 0:
                original_img = sitk_image[0]
                denoised_sitk.SetSpacing(original_img.GetSpacing())
                denoised_sitk.SetOrigin(original_img.GetOrigin())
                denoised_sitk.SetDirection(original_img.GetDirection())
            
            sitk.WriteImage(denoised_sitk, str(output_path))
            
            print(f"  ✓ Saved: {output_filename} ({D} slices)")
    
    print(f"\n{'='*80}")
    print("Denoising completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='NAFNet Denoising Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--pairs_csv', type=str, required=True,
                       help='Path to pairs.csv containing NC volume paths')
    parser.add_argument('--root_dir', type=str, default='E:\\LD-CT SR',
                       help='Root directory')
    parser.add_argument('--output_dir', type=str, default='Data/denoised_nc',
                       help='Directory to save denoised volumes')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Device 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 로드
    model = load_model(args.checkpoint, device)
    
    # 데이터셋 및 데이터로더 생성
    dataset = InferenceDataset(args.pairs_csv, args.root_dir)
    
    # Batch size는 1로 고정 (volumes은 크기가 다를 수 있음)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # SimpleITK는 multiprocessing과 충돌 가능
        pin_memory=False
    )
    
    # Inference 수행
    output_dir = Path(args.root_dir) / args.output_dir
    inference(model, dataloader, output_dir, device)


if __name__ == '__main__':
    main()