"""
3D UNet + Transformer Hybrid for CT Denoising

Architecture:
    - Encoder: 3D Conv blocks with residual connections
    - Bottleneck: 3D Swin Transformer blocks for long-range dependency
    - Decoder: 3D Conv blocks with skip connections
    - Input: (B, 1, D, H, W) where D=5 slices
    - Output: (B, 1, 1, H, W) center slice only (memory efficient)

Key advantages over 2D:
    - True 3D spatial context (not just channel stacking)
    - 3D convolutions capture inter-slice structure
    - 3D attention models z-axis dependency
    - Superior noise/structure separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import math


class Conv3DBlock(nn.Module):
    """3D Convolution block with GroupNorm and activation"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock3D(nn.Module):
    """3D Residual block"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = Conv3DBlock(channels, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, 1, 1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm(self.conv2(out))
        return out + residual


class WindowAttention3D(nn.Module):
    """
    3D Window-based Multi-head Self Attention
    Adapted from Swin Transformer for 3D medical imaging
    """
    def __init__(self, dim: int, window_size: Tuple[int, int, int], num_heads: int):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (D, H, W)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.view(B, D * H * W, C)
        
        # QKV projection
        qkv = self.qkv(x_flat).reshape(B, D * H * W, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)  # (B, heads, N, dim)
        out = out.transpose(1, 2).reshape(B, D * H * W, C)
        
        out = self.proj(out)
        out = out.view(B, D, H, W, C)
        return out


class TransformerBlock3D(nn.Module):
    """
    Simplified Swin-like Transformer block for 3D volumes
    
    Operates on full volume (no window shift) for simplicity,
    but still models global 3D dependency.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size=(5, 8, 8), num_heads=num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            x: (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # (B, C, D, H, W) → (B, D, H, W, C) for attention
        x_perm = x.permute(0, 2, 3, 4, 1)
        
        # Attention
        shortcut = x_perm
        x_perm = self.norm1(x_perm)
        x_perm = self.attn(x_perm)
        x_perm = shortcut + x_perm
        
        # MLP
        shortcut2 = x_perm
        x_perm = self.norm2(x_perm)
        x_perm = self.mlp(x_perm)
        x_perm = shortcut2 + x_perm
        
        # Back to (B, C, D, H, W)
        x_out = x_perm.permute(0, 4, 1, 2, 3).contiguous()
        return x_out


class UNet3DTransformer(nn.Module):
    """
    3D UNet with Transformer bottleneck for CT denoising
    
    Input:
        x: (B, 1, D, H, W) with D=5 slices
    
    Architecture:
        Encoder: 4 levels of 3D conv with downsampling
        Bottleneck: 2 Transformer blocks for global context
        Decoder: 4 levels of 3D conv with upsampling + skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_heads: int = 4,
    ):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            Conv3DBlock(in_channels, base_channels),
            ResBlock3D(base_channels),
        )
        # ↓↓↓ D는 그대로, H/W만 절반으로 (1,2,2) ↓↓↓
        self.down1 = nn.Conv3d(base_channels, base_channels * 2, 3, stride=(1, 2, 2), padding=1)
        
        self.enc2 = nn.Sequential(
            Conv3DBlock(base_channels * 2, base_channels * 2),
            ResBlock3D(base_channels * 2),
        )
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 4, 3, stride=(1, 2, 2), padding=1)
        
        self.enc3 = nn.Sequential(
            Conv3DBlock(base_channels * 4, base_channels * 4),
            ResBlock3D(base_channels * 4),
        )
        self.down3 = nn.Conv3d(base_channels * 4, base_channels * 8, 3, stride=(1, 2, 2), padding=1)
        
        self.enc4 = nn.Sequential(
            Conv3DBlock(base_channels * 8, base_channels * 8),
            ResBlock3D(base_channels * 8),
        )
        
        # Bottleneck: Transformer blocks
        self.transformer = nn.Sequential(
            TransformerBlock3D(base_channels * 8, num_heads),
            TransformerBlock3D(base_channels * 8, num_heads),
        )
        
        # Decoder
        # ↑↑↑ D 유지, H/W만 2배 (kernel=(1,2,2), stride=(1,2,2)) ↑↑↑
        self.up3 = nn.ConvTranspose3d(
            base_channels * 8, base_channels * 4,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec3 = nn.Sequential(
            Conv3DBlock(base_channels * 8, base_channels * 4),  # 8 = 4 (up) + 4 (skip)
            ResBlock3D(base_channels * 4),
        )
        
        self.up2 = nn.ConvTranspose3d(
            base_channels * 4, base_channels * 2,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec2 = nn.Sequential(
            Conv3DBlock(base_channels * 4, base_channels * 2),  # 4 = 2 (up) + 2 (skip)
            ResBlock3D(base_channels * 2),
        )
        
        self.up1 = nn.ConvTranspose3d(
            base_channels * 2, base_channels,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = nn.Sequential(
            Conv3DBlock(base_channels * 2, base_channels),  # 2 = 1 (up) + 1 (skip)
            ResBlock3D(base_channels),
        )
        
        # Final output: extract center slice
        self.out_conv = nn.Conv3d(base_channels, 1, 3, padding=1)
        self.out_activation = nn.Tanh()  # Limit noise to [-1, 1] range
    
    def forward(self, x):
        """
        Residual Learning: Predict NOISE, then subtract from input
        
        Args:
            x: (B, 1, D, H, W) where D=5
        Returns:
            denoised: (B, 1, 1, H, W)
            noise_center: (B, 1, 1, H, W) - Predicted noise for supervision
        """
        # Encoder
        e1 = self.enc1(x)           # (B, 32, D, H, W)
        e2 = self.enc2(self.down1(e1))  # (B, 64, D, H/2, W/2)
        e3 = self.enc3(self.down2(e2))  # (B, 128, D, H/4, W/4)
        e4 = self.enc4(self.down3(e3))  # (B, 256, D, H/8, W/8)
        
        # Bottleneck with Transformer
        b = self.transformer(e4)    # (B, 256, D, H/8, W/8)
        
        # Decoder with skip connections
        d3 = self.up3(b)            # (B, 128, D, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 256, D, H/4, W/4)
        d3 = self.dec3(d3)          # (B, 128, D, H/4, W/4)
        
        d2 = self.up2(d3)           # (B, 64, D, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 128, D, H/2, W/2)
        d2 = self.dec2(d2)          # (B, 64, D, H/2, W/2)
        
        d1 = self.up1(d2)           # (B, 32, D, H, W)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 64, D, H, W)
        d1 = self.dec1(d1)          # (B, 32, D, H, W)
        
        # Predict NOISE map with limited range
        noise_map = self.out_conv(d1)  # (B, 1, D, H, W)
        noise_map = self.out_activation(noise_map) * 0.3  # Increased for aggressive denoising
        
        # Extract center slice from noise and input
        D_dim = noise_map.shape[2]
        center_idx = D_dim // 2
        noise_center = noise_map[:, :, center_idx:center_idx+1, :, :]  # (B, 1, 1, H, W)
        input_center = x[:, :, center_idx:center_idx+1, :, :]  # (B, 1, 1, H, W)
        
        # Residual: Clean = Input - Noise
        denoised = input_center - noise_center
        denoised = torch.clamp(denoised, 0.0, 1.0)
        
        return denoised, noise_center


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity test
    model = UNet3DTransformer(in_channels=1, base_channels=32, num_heads=4)
    x = torch.randn(2, 1, 5, 128, 128)
    
    print("="*60)
    print("3D UNet + Transformer Hybrid")
    print("="*60)
    print(f"Input shape:  {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    print(f"Parameters:   {count_parameters(model):,}")
    print("="*60)
    
    # Expected output: (2, 1, 1, 128, 128) - center slice only
    assert out.shape == (2, 1, 1, 128, 128), f"Unexpected output shape: {out.shape}"
    print("✓ Model test passed")