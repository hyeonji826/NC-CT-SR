"""
model.py - Swin-N2NLDM 네트워크 구조
Swin Transformer U-Net 디노이저 + VAE 잠재공간 + Latent Diffusion Model.
Noise2Noise 방식으로 인접 슬라이스 쌍을 활용한 저선량 CT 디노이징.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# ============================================================
# Swin Transformer 기본 블록
# ============================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    (B, H, W, C) → (num_windows*B, window_size, window_size, C)
    로컬 윈도우로 분할하여 윈도우 내 주의집중을 수행하기 위한 전처리.
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int) -> torch.Tensor:
    """
    (num_windows*B, window_size, window_size, C) → (B, H, W, C)
    윈도우 분할을 원래 공간 배치로 복원.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    로컬 윈도우 내 Multi-Head Self-Attention.
    상대 위치 바이어스(Relative Position Bias)를 통해
    CT 영상의 미세한 공간 관계를 학습한다.
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 상대 위치 바이어스 테이블
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 상대 위치 인덱스 계산
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = torch.flatten(coords, 1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # 상대 위치 바이어스 적용
        ws = self.window_size
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws * ws, ws * ws, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        # Shifted window 어텐션 마스크
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer 블록: Window Attention + Shifted Window Attention.
    CT 영상의 노이즈 패턴(고주파)과 금속 아티팩트의 장거리 의존성을
    로컬/쉬프트 윈도우 메커니즘으로 동시에 학습한다.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def _compute_attn_mask(self, H: int, W: int,
                           device: torch.device,
                           shift: Optional[int] = None) -> Optional[torch.Tensor]:
        """Shifted window용 어텐션 마스크 계산."""
        ss = shift if shift is not None else self.shift_size
        if ss == 0:
            return None

        ws = self.window_size
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -ws), slice(-ws, -ss),
                    slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss),
                    slice(-ss, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        ws = self.window_size

        # 공간 해상도가 window_size보다 작을 경우 패딩 처리
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # Shift size를 현재 해상도에 맞게 조정
        effective_shift = min(self.shift_size, Hp // 2, Wp // 2)

        # Cyclic shift
        if effective_shift > 0:
            shifted = torch.roll(x, shifts=(-effective_shift, -effective_shift),
                                 dims=(1, 2))
        else:
            shifted = x

        # Window partition → Attention → Window reverse
        x_windows = window_partition(shifted, ws)
        x_windows = x_windows.view(-1, ws * ws, C)

        attn_mask = self._compute_attn_mask(Hp, Wp, x.device, shift=effective_shift) \
            if effective_shift > 0 else None
        attn_out = self.attn(x_windows, mask=attn_mask)

        attn_out = attn_out.view(-1, ws, ws, C)
        shifted = window_reverse(attn_out, ws, Hp, Wp)

        # Reverse cyclic shift
        if effective_shift > 0:
            x = torch.roll(shifted, shifts=(effective_shift, effective_shift),
                           dims=(1, 2))
        else:
            x = shifted

        # 패딩 제거
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + x

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SwinLayer(nn.Module):
    """여러 Swin Transformer 블록을 쌓은 하나의 스테이지."""

    def __init__(self, dim: int, depth: int, num_heads: int,
                 window_size: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


# ============================================================
# Patch 연산: Embed, Merging (다운샘플), Expanding (업샘플)
# ============================================================

class PatchEmbed(nn.Module):
    """이미지를 패치로 분할하고 임베딩 차원으로 변환."""

    def __init__(self, patch_size: int = 4, in_chans: int = 1,
                 embed_dim: int = 96):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)                           # (B, C, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, H*W, C)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """공간 해상도를 절반으로 줄이고 채널을 2배로 확장 (다운샘플링)."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int
                ) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)    # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)                       # (B, H/2*W/2, 2C)
        return x, H // 2, W // 2


class PatchExpanding(nn.Module):
    """공간 해상도를 2배로 확장하고 채널을 절반으로 축소 (업샘플링)."""

    def __init__(self, dim: int):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x: torch.Tensor, H: int, W: int
                ) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        x = self.expand(x)                          # (B, L, 2C)
        x = x.view(B, H, W, 2 * C)
        # Pixel-shuffle 방식 재배열: (B, H, W, 2C) → (B, 2H, 2W, C/2)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, C // 2)
        x = self.norm(x)
        x = x.view(B, -1, C // 2)
        return x, 2 * H, 2 * W


# ============================================================
# Swin-UNet Generator (Encoder-Decoder with Skip Connections)
# ============================================================

class SwinUNetGenerator(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 96,
                 depths: Tuple[int, ...] = (2, 2, 2),
                 bottleneck_depth: int = 6,
                 num_heads: Tuple[int, ...] = (3, 6, 12),
                 bottleneck_heads: int = 24,
                 window_size: int = 8, patch_size: int = 4,
                 residual_learning: bool = False,
                 residual_scale: float = 0.5):
        super().__init__()
        self.patch_size = patch_size
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.residual_learning = residual_learning
        self.residual_scale = residual_scale

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        for i in range(self.num_stages):
            dim = embed_dim * (2 ** i)
            self.encoder_layers.append(SwinLayer(dim, depths[i], num_heads[i], window_size))
            self.downsample_layers.append(PatchMerging(dim))

        # Bottleneck
        bottleneck_dim = embed_dim * (2 ** self.num_stages)
        self.bottleneck = SwinLayer(bottleneck_dim, bottleneck_depth, bottleneck_heads, window_size)

        # Decoder
        self.upsample_layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in reversed(range(self.num_stages)):
            dim_up = embed_dim * (2 ** (i + 1))
            dim_out = embed_dim * (2 ** i)
            self.upsample_layers.append(PatchExpanding(dim_up))
            self.skip_projections.append(nn.Linear(dim_out * 2, dim_out))
            self.decoder_layers.append(SwinLayer(dim_out, depths[i], num_heads[i], window_size))

        # [수정] 최종 정제부: Bilinear Upsampling 후 Conv로 선명도 보강
        self.final_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(embed_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(embed_dim // 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_proj = nn.Conv2d(embed_dim // 2, in_chans, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H_orig, W_orig = x.shape
        identity_input = x  # 잔차 학습용 원본 보존

        # 패딩 처리
        pad_h = (self.patch_size * 2**self.num_stages - H_orig % (self.patch_size * 2**self.num_stages)) % (self.patch_size * 2**self.num_stages)
        pad_w = (self.patch_size * 2**self.num_stages - W_orig % (self.patch_size * 2**self.num_stages)) % (self.patch_size * 2**self.num_stages)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        
        # Patch Embedding
        x, H, W = self.patch_embed(x)

        # Encoder
        skips = []
        for enc, down in zip(self.encoder_layers, self.downsample_layers):
            x = enc(x, H, W)
            skips.append((x, H, W))
            x, H, W = down(x, H, W)

        x = self.bottleneck(x, H, W)

        # Decoder
        for up, skip_proj, dec in zip(self.upsample_layers, self.skip_projections, self.decoder_layers):
            x, H, W = up(x, H, W)
            skip_x, sH, sW = skips.pop()
            x = torch.cat([x, skip_x], dim=-1)
            x = skip_proj(x)
            x = dec(x, H, W)

        # [수정] 차원 복원 및 최종 업샘플링
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)
        x = F.interpolate(x, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        x = self.final_upsample(x)
        x = self.final_proj(x)

        if self.residual_learning:
            residual = torch.tanh(x) * self.residual_scale
            return torch.clamp(identity_input + residual, -1.0, 1.0)
        return torch.clamp(x, -1.0, 1.0)


# ============================================================
# VAE (Variational Autoencoder) — 잠재 공간 설계
# ============================================================

class VAEEncoder(nn.Module):
    """고해상도 CT를 잠재 공간으로 압축하여 연산 효율을 극대화."""

    def __init__(self, in_chans: int = 1, base_ch: int = 64,
                 latent_ch: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, base_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # mu, logvar 각각 출력
        self.conv_mu = nn.Conv2d(base_ch * 8, latent_ch, 3, stride=1, padding=1)
        self.conv_logvar = nn.Conv2d(base_ch * 8, latent_ch, 3, stride=1, padding=1)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        return mu, logvar


class VAEDecoder(nn.Module):
    """잠재 공간에서 원본 해상도로 복원."""

    def __init__(self, out_chans: int = 1, base_ch: int = 64,
                 latent_ch: int = 4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, base_ch * 8, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, out_chans, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class VAE(nn.Module):
    """
    Variational Autoencoder: 영상을 잠재 공간으로 압축/복원.
    Reparameterization trick으로 미분 가능한 샘플링.
    """

    def __init__(self, in_chans: int = 1, base_ch: int = 64,
                 latent_ch: int = 4):
        super().__init__()
        self.encoder = VAEEncoder(in_chans, base_ch, latent_ch)
        self.decoder = VAEDecoder(in_chans, base_ch, latent_ch)

    def reparameterize(self, mu: torch.Tensor,
                       logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization Trick: z = mu + sigma * epsilon"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ============================================================
# Latent Diffusion Model (LDM)
# ============================================================

class SinusoidalPosEmbed(nn.Module):
    """확산 타임스텝을 위한 사인/코사인 위치 임베딩."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    """타임스텝 조건화 Residual Block."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """잠재 공간 내 Self-Attention."""

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)     # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


class LatentUNet(nn.Module):
    """
    잠재 공간 내에서 노이즈를 예측하는 UNet.
    타임스텝 조건화 + Self-Attention으로
    확산 과정의 가우시안 노이즈를 단계적으로 제거한다.
    """

    def __init__(self, latent_ch: int = 4, base_ch: int = 128,
                 ch_mult: Tuple[int, ...] = (1, 2, 4),
                 time_dim: int = 256, num_heads: int = 4):
        super().__init__()

        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc_conv_in = nn.Conv2d(latent_ch, base_ch, 3, padding=1)
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()
        self.enc_attns = nn.ModuleList()

        chs = [base_ch]
        ch = base_ch
        for mult in ch_mult:
            out_ch = base_ch * mult
            self.enc_blocks.append(ResBlock(ch, out_ch, time_dim))
            self.enc_attns.append(SelfAttention2d(out_ch, num_heads))
            self.enc_downs.append(nn.Conv2d(out_ch, out_ch, 4, 2, 1))
            ch = out_ch
            chs.append(ch)

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = SelfAttention2d(ch, num_heads)
        self.mid_block2 = ResBlock(ch, ch, time_dim)

        # Decoder
        self.dec_blocks = nn.ModuleList()
        self.dec_ups = nn.ModuleList()
        self.dec_attns = nn.ModuleList()

        for mult in reversed(ch_mult):
            out_ch = base_ch * mult
            self.dec_ups.append(
                nn.ConvTranspose2d(ch, ch, 4, 2, 1)
            )
            self.dec_blocks.append(ResBlock(ch + chs.pop(), out_ch, time_dim))
            self.dec_attns.append(SelfAttention2d(out_ch, num_heads))
            ch = out_ch

        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, latent_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        # Encoder
        h = self.enc_conv_in(x)
        skips = [h]
        for blk, attn, down in zip(self.enc_blocks, self.enc_attns, self.enc_downs):
            h = blk(h, t_emb)
            h = attn(h)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for blk, attn, up in zip(self.dec_blocks, self.dec_attns, self.dec_ups):
            h = up(h)
            h = torch.cat([h, skips.pop()], dim=1)
            h = blk(h, t_emb)
            h = attn(h)

        return self.out_conv(h)


class GaussianDiffusion(nn.Module):
    """
    잠재 공간 내 가우시안 확산 프로세스.
    Forward: 점진적 노이즈 추가 q(x_t | x_0)
    Reverse: 학습된 UNet으로 노이즈 예측 후 단계적 제거
    """

    def __init__(self, denoise_net: nn.Module,
                 timesteps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        super().__init__()
        self.denoise_net = denoise_net
        self.timesteps = timesteps

        # 선형 베타 스케줄
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',
                             torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas',
                             torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance',
                             betas * (1.0 - F.pad(alphas_cumprod[:-1], (1, 0), value=1.0))
                             / (1.0 - alphas_cumprod))

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward process: x_0에 t 단계까지 노이즈 추가."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x_0 + sqrt_omac * noise

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """확산 손실: 랜덤 타임스텝의 노이즈 예측 MSE."""
        B = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted = self.denoise_net(x_t, t)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """Reverse process: 단일 타임스텝 역방향 샘플링."""
        B = x_t.shape[0]
        t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        predicted_noise = self.denoise_net(x_t, t_batch)

        sqrt_recip = self.sqrt_recip_alphas[t]
        beta = self.betas[t]
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[t]

        mean = sqrt_recip * (x_t - beta / sqrt_omac * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...],
               device: torch.device) -> torch.Tensor:
        """전체 역방향 프로세스: 순수 노이즈 → 디노이즈된 잠재 벡터."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, t)
        return x


# ============================================================
# Swin-N2NLDM 통합 모델
# ============================================================

class SwinN2NLDM(nn.Module):
    """
    Swin Transformer U-Net + VAE + Latent Diffusion Model 통합.
    Noise2Noise 방식: 인접 슬라이스 쌍으로 디노이저를 학습.

    Components:
        - denoiser: Swin U-Net (노이즈 제거)
        - VAE: 잠재 공간 압축/복원
        - LDM: 잠재 공간 내 확산 기반 정밀 복원
    """

    def __init__(self,
                 in_chans: int = 1,
                 embed_dim: int = 96,
                 depths: Tuple[int, ...] = (2, 2, 2),
                 bottleneck_depth: int = 6,
                 num_heads: Tuple[int, ...] = (3, 6, 12),
                 bottleneck_heads: int = 24,
                 window_size: int = 8,
                 patch_size: int = 4,
                 vae_base_ch: int = 64,
                 latent_ch: int = 4,
                 diffusion_steps: int = 1000,
                 residual_learning: bool = False,
                 residual_scale: float = 0.5):
        super().__init__()

        # 디노이저: Swin Transformer U-Net
        self.denoiser = SwinUNetGenerator(
            in_chans=in_chans, embed_dim=embed_dim, depths=depths,
            bottleneck_depth=bottleneck_depth, num_heads=num_heads,
            bottleneck_heads=bottleneck_heads, window_size=window_size,
            patch_size=patch_size,
            residual_learning=residual_learning,
            residual_scale=residual_scale
        )

        # VAE: 잠재 공간 압축
        self.vae = VAE(in_chans, vae_base_ch, latent_ch)

        # LDM: 잠재 공간 내 확산 모델
        latent_unet = LatentUNet(latent_ch=latent_ch)
        self.ldm = GaussianDiffusion(latent_unet, timesteps=diffusion_steps)

    def forward(self, noisy_input: torch.Tensor) -> torch.Tensor:
        """학습용: 노이즈 입력 → 디노이징 출력."""
        return self.denoiser(noisy_input)

    def forward_vae(self, x: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE 순전파: 인코딩 → 재매개변수화 → 디코딩."""
        return self.vae(x)

    def forward_diffusion_loss(self, x: torch.Tensor) -> torch.Tensor:
        """LDM 학습 손실: VAE 잠재 벡터의 노이즈 예측 오차."""
        with torch.no_grad():
            mu, logvar = self.vae.encode(x)
            z = self.vae.reparameterize(mu, logvar)
        return self.ldm.compute_loss(z)

    @torch.no_grad()
    def inference(self, noisy_input: torch.Tensor,
                  use_diffusion: bool = True,
                  refine_steps: int = 20) -> torch.Tensor:
        """
        추론 파이프라인:
        1. Swin U-Net 디노이저로 초기 복원
        2. (선택) VAE 인코딩 → 확산 정제 → VAE 디코딩
        """
        self.eval()

        # Step 1: 디노이저로 초기 복원
        clean_est = self.denoiser(noisy_input)

        if not use_diffusion:
            return clean_est

        # Step 2: VAE 잠재 공간에서 확산 기반 정제
        mu, _ = self.vae.encode(clean_est)
        z = mu  # 추론 시 deterministic

        t_start = min(refine_steps, self.ldm.timesteps)
        noise = torch.randn_like(z)
        t_tensor = torch.full((z.shape[0],), t_start - 1,
                              device=z.device, dtype=torch.long)
        z_noisy = self.ldm.q_sample(z, t_tensor, noise)

        for t in reversed(range(t_start)):
            z_noisy = self.ldm.p_sample(z_noisy, t)

        refined = self.vae.decode(z_noisy)
        return refined


# ============================================================
# Artifact Removal Network
# ============================================================

class HUGuidedClassifier(nn.Module):
    """
    HU 값 기반 조직 타입 분류기 (Soft Segmentation).

    조직 타입 (6종):
        0. Air/Gas: -1000 ~ -500 HU
        1. Fat: -200 ~ -60 HU
        2. Water/Fluid: -10 ~ 40 HU (위액, 방광, 낭종, 병변 후보)
        3. Soft Tissue: 30 ~ 80 HU (간, 신장, 비장, 종양)
        4. Vessel: 100 ~ 200 HU (비조영 혈관)
        5. Bone: 300+ HU

    각 픽셀이 여러 조직 타입에 동시 소속 가능 (Soft probability).
    """

    def __init__(self, num_tissue_types: int = 6):
        super().__init__()
        self.num_types = num_tissue_types

        # HU 범위별 Gaussian membership parameters
        # (mean, std)
        self.register_buffer('tissue_means', torch.tensor([
            -750.0,  # Air
            -120.0,  # Fat
            15.0,    # Water/Fluid
            55.0,    # Soft Tissue
            150.0,   # Vessel
            600.0    # Bone
        ]))
        self.register_buffer('tissue_stds', torch.tensor([
            300.0,   # Air (wide range)
            40.0,    # Fat
            30.0,    # Water (핵심! 저음영 영역)
            25.0,    # Soft Tissue
            50.0,    # Vessel
            400.0    # Bone (wide range)
        ]))

        # Learnable refinement (spatial context 활용)
        self.refine_conv = nn.Sequential(
            nn.Conv2d(num_tissue_types, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, num_tissue_types, 3, padding=1),
            nn.Softmax(dim=1)  # Soft probability
        )

    def gaussian_membership(self, hu: torch.Tensor,
                             mean: torch.Tensor,
                             std: torch.Tensor) -> torch.Tensor:
        """
        Gaussian membership function.

        Args:
            hu: (B, 1, H, W) HU values
            mean: (1,) Scalar
            std: (1,) Scalar
        Returns:
            prob: (B, 1, H, W) Membership probability
        """
        return torch.exp(-((hu - mean) ** 2) / (2 * std ** 2))

    def forward(self, img_hu: torch.Tensor) -> torch.Tensor:
        """
        HU-guided tissue classification.

        Args:
            img_hu: (B, 1, H, W) HU values [-1000, 1000]

        Returns:
            tissue_prob: (B, 6, H, W) Soft segmentation
                - Channel 0: Air
                - Channel 1: Fat
                - Channel 2: Water/Fluid (저음영!)
                - Channel 3: Soft Tissue
                - Channel 4: Vessel
                - Channel 5: Bone
        """
        B, C, H, W = img_hu.shape

        # Compute membership for each tissue type
        tissue_probs = []
        for i in range(self.num_types):
            prob = self.gaussian_membership(
                img_hu,
                self.tissue_means[i],
                self.tissue_stds[i]
            )
            tissue_probs.append(prob)

        tissue_probs = torch.cat(tissue_probs, dim=1)  # (B, 6, H, W)

        # Normalize (sum to 1)
        tissue_probs = tissue_probs / (tissue_probs.sum(dim=1, keepdim=True) + 1e-8)

        # Learnable refinement (spatial context)
        refined = self.refine_conv(tissue_probs)

        return refined


class SpatialCoherenceAnalyzer(nn.Module):
    """
    공간적 연속성 분석기.

    핵심: 노이즈 vs 병변 vs 저음영(물) 구분
        - 노이즈: 무작위, 불연속, 낮은 coherence
        - 병변: 연속적, 경계 명확, 중간 coherence
        - 물(위액): 넓은 영역, 매우 높은 coherence

    Output:
        - Local variance (노이즈 지표)
        - Gradient magnitude (경계 지표)
        - Non-local similarity (구조 지표)
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()

        # Feature extractor for non-local similarity
        self.feature_net = nn.Sequential(
            nn.Conv2d(1, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def compute_local_variance(self, x: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        """
        지역 분산 계산 (노이즈 지표).

        높은 분산 = 노이즈 또는 텍스처
        """
        kernel = torch.ones(1, 1, kernel_size, kernel_size,
                            device=x.device) / (kernel_size ** 2)
        pad = kernel_size // 2

        mean = F.conv2d(x, kernel, padding=pad)
        mean_sq = F.conv2d(x ** 2, kernel, padding=pad)
        var = mean_sq - mean ** 2

        return var

    def compute_gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gradient magnitude (경계 지표).

        높은 gradient = 경계 (장기, 병변, 혈관)
        """
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        return grad_mag

    def compute_nonlocal_similarity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-local self-similarity (구조 지표) - Memory-efficient version.

        높은 유사도 = 구조적 (병변, 물, 장기)
        낮은 유사도 = 무작위 (노이즈)

        메모리 절약: 32×32로 다운샘플링 후 similarity 계산
        """
        B, C, H, W = x.shape

        # 1. 다운샘플링 (512×512 → 32×32)
        x_down = F.adaptive_avg_pool2d(x, (32, 32))  # (B, 1, 32, 32)

        # 2. Feature extraction
        features = self.feature_net(x_down)  # (B, C, 32, 32)
        B, C, H_down, W_down = features.shape

        # 3. Flatten spatial dimensions (32×32 = 1024 pixels)
        feat_flat = features.view(B, C, -1)  # (B, C, 1024)

        # 4. Self-attention similarity matrix (1024×1024 = 4MB per batch - manageable!)
        similarity = torch.bmm(feat_flat.transpose(1, 2), feat_flat)  # (B, 1024, 1024)
        similarity = F.softmax(similarity / math.sqrt(C), dim=-1)

        # 5. Average similarity per pixel (coherence score)
        avg_sim = similarity.mean(dim=-1, keepdim=True)  # (B, 1024, 1)
        avg_sim = avg_sim.view(B, 1, H_down, W_down)  # (B, 1, 32, 32)

        # 6. 원래 해상도로 업샘플링
        avg_sim = F.interpolate(avg_sim, size=(H, W), mode='bilinear', align_corners=False)

        return avg_sim

    def forward(self, x_hu: torch.Tensor) -> torch.Tensor:
        """
        공간 coherence 분석.

        Args:
            x_hu: (B, 1, H, W) HU values

        Returns:
            coherence_map: (B, 3, H, W)
                - Channel 0: Local variance (노이즈 지표)
                - Channel 1: Gradient magnitude (경계 지표)
                - Channel 2: Non-local similarity (구조 지표)
        """
        local_var = self.compute_local_variance(x_hu)          # (B, 1, H, W)
        grad_mag = self.compute_gradient_magnitude(x_hu)       # (B, 1, H, W)
        nonlocal_sim = self.compute_nonlocal_similarity(x_hu)  # (B, 1, H, W)

        coherence_map = torch.cat([local_var, grad_mag, nonlocal_sim], dim=1)  # (B, 3, H, W)

        return coherence_map


class WaveletFrequencyDecomposer(nn.Module):
    """
    Haar Wavelet 기반 주파수 분해.

    4개 Sub-band:
        - LL (Low-Low): 저주파 → 구조, 병변, 장기
        - LH (Low-High): 수평 경계 → 장기 경계
        - HL (High-Low): 수직 경계 → 혈관
        - HH (High-High): 대각 고주파 → 노이즈 주 성분!

    각 sub-band별 처리 전략:
        - LL: 약한 denoising (구조 보존)
        - LH, HL: 보존 (경계 보존)
        - HH: 강한 denoising (노이즈 제거)
    """

    def __init__(self):
        super().__init__()

        # Haar wavelet kernels
        self._init_haar_kernels()

        # Sub-band별 처리 네트워크
        self.ll_process = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
        )

        self.lh_hl_process = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 2, 3, padding=1),
        )

        self.hh_process = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),  # Attenuation factor [0, 1]
        )

    def _init_haar_kernels(self):
        """Haar wavelet decomposition/reconstruction kernels."""
        # Low-pass
        h0 = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        # High-pass
        h1 = torch.tensor([[0.5, -0.5]], dtype=torch.float32)

        # 2D kernels (separable)
        LL = (h0.T @ h0).unsqueeze(0).unsqueeze(0)  # (1, 1, 2, 2)
        LH = (h0.T @ h1).unsqueeze(0).unsqueeze(0)
        HL = (h1.T @ h0).unsqueeze(0).unsqueeze(0)
        HH = (h1.T @ h1).unsqueeze(0).unsqueeze(0)

        self.register_buffer('kernel_LL', LL)
        self.register_buffer('kernel_LH', LH)
        self.register_buffer('kernel_HL', HL)
        self.register_buffer('kernel_HH', HH)

        # Reconstruction (transpose)
        self.register_buffer('kernel_LL_inv', LL * 4)
        self.register_buffer('kernel_LH_inv', LH * 4)
        self.register_buffer('kernel_HL_inv', HL * 4)
        self.register_buffer('kernel_HH_inv', HH * 4)

    def dwt2(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        2D Discrete Wavelet Transform (Haar).

        Args:
            x: (B, 1, H, W)
        Returns:
            LL, LH, HL, HH: Each (B, 1, H//2, W//2)
        """
        LL = F.conv2d(x, self.kernel_LL, stride=2)
        LH = F.conv2d(x, self.kernel_LH, stride=2)
        HL = F.conv2d(x, self.kernel_HL, stride=2)
        HH = F.conv2d(x, self.kernel_HH, stride=2)

        return LL, LH, HL, HH

    def idwt2(self, LL: torch.Tensor, LH: torch.Tensor,
              HL: torch.Tensor, HH: torch.Tensor) -> torch.Tensor:
        """
        Inverse 2D DWT (Reconstruction).

        Args:
            LL, LH, HL, HH: Each (B, 1, H//2, W//2)
        Returns:
            x: (B, 1, H, W)
        """
        LL_up = F.conv_transpose2d(LL, self.kernel_LL_inv, stride=2)
        LH_up = F.conv_transpose2d(LH, self.kernel_LH_inv, stride=2)
        HL_up = F.conv_transpose2d(HL, self.kernel_HL_inv, stride=2)
        HH_up = F.conv_transpose2d(HH, self.kernel_HH_inv, stride=2)

        x = LL_up + LH_up + HL_up + HH_up

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wavelet decomposition → processing → reconstruction.

        Args:
            x: (B, 1, H, W) [-1, 1] normalized

        Returns:
            freq_features: (B, 4, H//2, W//2) - 4-band features for fusion
            denoised: (B, 1, H, W) - Wavelet-denoised output
        """
        # Decomposition
        LL, LH, HL, HH = self.dwt2(x)

        # 각 sub-band별 처리
        LL_proc = self.ll_process(LL)  # 저주파 약한 처리

        LH_HL = torch.cat([LH, HL], dim=1)
        LH_HL_proc = self.lh_hl_process(LH_HL)  # 경계 보존
        LH_proc, HL_proc = torch.chunk(LH_HL_proc, 2, dim=1)

        HH_attenuation = self.hh_process(HH)  # 노이즈 감쇠 factor
        HH_proc = HH * HH_attenuation  # Element-wise

        # Frequency features (for fusion module)
        freq_features = torch.cat([LL, LH, HL, HH], dim=1)  # (B, 4, H//2, W//2)

        # Reconstruction
        denoised = self.idwt2(LL_proc, LH_proc, HL_proc, HH_proc)

        return freq_features, denoised


class AdaptiveFeatureFusion(nn.Module):
    """
    HU + Spatial + Frequency 특징 융합.

    Cross-attention으로 다중 모달 특징 통합.
    """

    def __init__(self, out_channels: int = 64):
        super().__init__()

        # Frequency feature upsampling
        self.freq_upsample = nn.Sequential(
            nn.ConvTranspose2d(4, 32, 2, stride=2),  # 4-band wavelet
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Projection layers
        self.hu_proj = nn.Conv2d(6, 32, 1)       # 6 tissue types
        self.spatial_proj = nn.Conv2d(3, 32, 1)  # 3 coherence maps
        self.freq_proj = nn.Conv2d(32, 32, 1)

        # Efficient channel-wise fusion (메모리 효율적)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(96, out_channels * 2, 3, padding=1),  # 32*3 = 96
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

    def forward(self,
                hu_mask: torch.Tensor,
                spatial_coh: torch.Tensor,
                freq_feat: torch.Tensor) -> torch.Tensor:
        """
        Multi-modal feature fusion (Memory-efficient).

        Args:
            hu_mask: (B, 6, H, W)
            spatial_coh: (B, 3, H, W)
            freq_feat: (B, 4, H//2, W//2)

        Returns:
            fused: (B, 64, H, W)
        """
        B, _, H, W = hu_mask.shape

        # Frequency upsampling
        freq_up = self.freq_upsample(freq_feat)  # (B, 32, H, W)

        # Projection
        hu_emb = self.hu_proj(hu_mask)           # (B, 32, H, W)
        spatial_emb = self.spatial_proj(spatial_coh)  # (B, 32, H, W)
        freq_emb = self.freq_proj(freq_up)       # (B, 32, H, W)

        # Direct concatenation + Conv fusion (메모리 효율적!)
        # Cross-attention 대신 Conv로 feature interaction 학습
        fused = torch.cat([hu_emb, spatial_emb, freq_emb], dim=1)  # (B, 96, H, W)
        fused = self.fusion_conv(fused)  # (B, 64, H, W)

        return fused


class SimpleGate(nn.Module):
    """NAFNet Simplified Channel Attention Gate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """NAFNet Simplified Channel Attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.avg_pool(x)
        attn = self.conv(attn)
        return attn


class NAFBlock(nn.Module):
    """
    NAFNet Block (Nonlinear Activation Free).

    특징:
        - No ReLU/GELU → 메모리 효율
        - SimpleGate activation
        - Channel attention
    """

    def __init__(self, channels: int, dw_expansion: int = 2, ffn_expansion: int = 2):
        super().__init__()

        dw_ch = channels * dw_expansion

        self.norm1 = nn.LayerNorm(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, 1)
        self.dw_conv = nn.Conv2d(dw_ch, dw_ch, 3, padding=1, groups=dw_ch)
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(dw_ch // 2, channels, 1)
        self.sca = SimplifiedChannelAttention(dw_ch // 2)

        # FFN
        self.norm2 = nn.LayerNorm(channels)
        ffn_ch = channels * ffn_expansion
        self.ffn_conv1 = nn.Conv2d(channels, ffn_ch, 1)
        self.ffn_conv2 = nn.Conv2d(ffn_ch // 2, channels, 1)

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise convolution branch
        B, C, H, W = x.shape
        y = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # LN

        y = self.conv1(y)
        y = self.dw_conv(y)
        y = self.sg(y)
        y = self.sca(y) * y
        y = self.conv2(y)

        x = x + y * self.beta

        # FFN
        y = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.ffn_conv1(y)
        y = self.sg(y)
        y = self.ffn_conv2(y)

        x = x + y * self.gamma

        return x


class HybridBackbone(nn.Module):
    """
    Hybrid Backbone: Swin (Encoder) + NAFNet (Bottleneck) + Swin (Decoder).

    구조:
        - Stage 1-2: Swin (Global context, long-range)
        - Stage 3 (Bottleneck): NAFNet (Efficient, deep)
        - Stage 4-5: Swin (Structure restoration)
    """

    def __init__(self,
                 in_channels: int = 64,
                 img_size: int = 512,
                 swin_embed_dim: int = 96,
                 swin_depths: Tuple[int, int] = (2, 2),
                 swin_num_heads: Tuple[int, int] = (3, 6),
                 nafnet_depth: int = 18,
                 nafnet_width: int = 384,
                 window_size: int = 8,
                 patch_size: int = 4):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, swin_embed_dim, 3, padding=1)

        # === Swin Encoder (Stage 1-2) ===
        self.swin_enc_embed = PatchEmbed(patch_size, swin_embed_dim, swin_embed_dim)
        self.swin_enc_layers = nn.ModuleList()
        self.swin_enc_downs = nn.ModuleList()

        for i in range(len(swin_depths)):
            dim = swin_embed_dim * (2 ** i)
            self.swin_enc_layers.append(
                SwinLayer(dim, swin_depths[i], swin_num_heads[i], window_size)
            )
            if i < len(swin_depths) - 1:
                self.swin_enc_downs.append(PatchMerging(dim))

        # === NAFNet Bottleneck ===
        bottleneck_dim = swin_embed_dim * (2 ** (len(swin_depths) - 1))
        self.bottleneck_proj_in = nn.Sequential(
            nn.Conv2d(bottleneck_dim, nafnet_width, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.nafnet_blocks = nn.Sequential(*[
            NAFBlock(nafnet_width) for _ in range(nafnet_depth)
        ])

        self.bottleneck_proj_out = nn.Sequential(
            nn.Conv2d(nafnet_width, bottleneck_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # === Swin Decoder (Stage 4-5) ===
        self.swin_dec_ups = nn.ModuleList()
        self.swin_dec_skips = nn.ModuleList()
        self.swin_dec_layers = nn.ModuleList()

        for i in reversed(range(len(swin_depths))):
            dim_up = swin_embed_dim * (2 ** (i + 1)) if i < len(swin_depths) - 1 else bottleneck_dim
            dim_out = swin_embed_dim * (2 ** i)

            if i < len(swin_depths) - 1:
                self.swin_dec_ups.append(PatchExpanding(dim_up))
                self.swin_dec_skips.append(nn.Linear(dim_out * 2, dim_out))
                self.swin_dec_layers.append(
                    SwinLayer(dim_out, swin_depths[i], swin_num_heads[i], window_size)
                )

        # Final output projection
        self.final_proj = nn.Sequential(
            nn.Conv2d(swin_embed_dim, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
        )

        # Residual learning
        self.residual_learning = True
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, input_orig: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 64, H, W) - Fused features
            input_orig: (B, 1, H, W) - Original noisy input (for residual)

        Returns:
            output: (B, 1, H, W) - Denoised CT
        """
        B, C, H_orig, W_orig = x.shape

        # Input projection
        x = self.input_proj(x)  # (B, embed_dim, H, W)

        # Padding for patch embedding
        pad_h = (self.patch_size - H_orig % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W_orig % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # === Swin Encoder ===
        x, H, W = self.swin_enc_embed(x)
        enc_skips = []

        for i, (enc_layer, down) in enumerate(zip(self.swin_enc_layers[:-1], self.swin_enc_downs)):
            x = enc_layer(x, H, W)
            enc_skips.append((x, H, W))
            x, H, W = down(x, H, W)

        # Last encoder layer
        x = self.swin_enc_layers[-1](x, H, W)

        # === NAFNet Bottleneck ===
        x = x.transpose(1, 2).view(B, -1, H, W)  # (B, C, H, W)
        x = self.bottleneck_proj_in(x)
        x = self.nafnet_blocks(x)
        x = self.bottleneck_proj_out(x)
        x = x.flatten(2).transpose(1, 2)  # (B, HW, C)

        # === Swin Decoder ===
        for up, skip_proj, dec_layer in zip(self.swin_dec_ups, self.swin_dec_skips, self.swin_dec_layers):
            x, H, W = up(x, H, W)
            skip_x, sH, sW = enc_skips.pop()
            x = torch.cat([x, skip_x], dim=-1)
            x = skip_proj(x)
            x = dec_layer(x, H, W)

        # === Output ===
        x = x.transpose(1, 2).view(B, -1, H, W)  # (B, C, H, W)
        x = F.interpolate(x, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        x = self.final_proj(x)  # (B, 1, H, W)

        # Residual learning
        if self.residual_learning:
            noise_residual = torch.tanh(x) * self.residual_scale
            output = torch.clamp(input_orig - noise_residual, -1.0, 1.0)
        else:
            output = torch.clamp(x, -1.0, 1.0)

        return output


class ArtifactRemovalNet(nn.Module):
    """
    Artifact Removal Network (CT Streak Artifact Removal).

    통합 아키텍처:
        1. Multi-Feature Extraction:
            - HU Classifier (조직 타입)
            - Spatial Coherence (연속성)
            - Wavelet Decomposition (주파수)

        2. Feature Fusion:
            - Cross-attention 기반 융합

        3. Hybrid Backbone:
            - Swin (Global) + NAFNet (Efficient) + Swin (Restore)

        4. Residual Learning:
            - Noise만 예측 → 빠른 수렴
    """

    def __init__(self,
                 img_size: int = 512,
                 swin_embed_dim: int = 96,
                 swin_depths: Tuple[int, int] = (2, 2),
                 swin_num_heads: Tuple[int, int] = (3, 6),
                 nafnet_depth: int = 18,
                 nafnet_width: int = 384,
                 use_streak_map: bool = False):
        super().__init__()

        # === Multi-Feature Extractors ===
        self.hu_classifier = HUGuidedClassifier(num_tissue_types=6)
        self.spatial_analyzer = SpatialCoherenceAnalyzer(embed_dim=32)
        self.wavelet_decomposer = WaveletFrequencyDecomposer()

        # === Feature Fusion ===
        self.feature_fusion = AdaptiveFeatureFusion(out_channels=64)

        # === Streak Map Conditioning (optional) ===
        self.use_streak_map = use_streak_map
        if use_streak_map:
            self.streak_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            backbone_in_channels = 64 + 16
        else:
            backbone_in_channels = 64

        # === Hybrid Backbone ===
        self.hybrid_backbone = HybridBackbone(
            in_channels=backbone_in_channels,
            img_size=img_size,
            swin_embed_dim=swin_embed_dim,
            swin_depths=swin_depths,
            swin_num_heads=swin_num_heads,
            nafnet_depth=nafnet_depth,
            nafnet_width=nafnet_width,
        )

    def forward(self, noisy_ct_normalized: torch.Tensor,
                noisy_ct_hu: torch.Tensor,
                streak_map: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass - Residual Learning (노이즈 예측).

        HybridBackbone이 내부적으로 residual learning을 수행하여
        denoised output을 직접 반환함. 여기서는 이중 차감하지 않음.

        Args:
            noisy_ct_normalized: (B, 1, H, W) [-1, 1] normalized
            noisy_ct_hu: (B, 1, H, W) HU values [-1000, 1000]

        Returns:
            denoised: (B, 1, H, W) [-1, 1] denoised output
            aux_outputs: Dict - Auxiliary outputs for loss/visualization
                - noise_pred: (B, 1, H, W) - 예측된 노이즈 (input - denoised)
                - hu_mask: (B, 6, H, W)
                - spatial_coherence: (B, 3, H, W)
                - wavelet_denoised: (B, 1, H, W)
                - water_mask: (B, 1, H, W)
                - bone_mask: (B, 1, H, W)
        """
        # === Multi-Feature Extraction ===
        hu_mask = self.hu_classifier(noisy_ct_hu)  # (B, 6, H, W)
        spatial_coh = self.spatial_analyzer(noisy_ct_hu)  # (B, 3, H, W)
        freq_feat, wavelet_denoised = self.wavelet_decomposer(noisy_ct_normalized)  # (B, 4, H//2, W//2), (B, 1, H, W)

        # === Extract tissue-specific masks for loss weighting ===
        water_mask = hu_mask[:, 2:3, :, :]  # Channel 2: Water/Fluid
        bone_mask = hu_mask[:, 5:6, :, :]   # Channel 5: Bone

        # === Feature Fusion ===
        fused_features = self.feature_fusion(hu_mask, spatial_coh, freq_feat)  # (B, 64, H, W)

        # === Streak Conditioning ===
        if self.use_streak_map and streak_map is not None:
            streak_feat = self.streak_encoder(streak_map)  # (B, 16, H, W)
            fused_features = torch.cat([fused_features, streak_feat], dim=1)  # (B, 80, H, W)
        elif self.use_streak_map:
            # streak_map 미제공 시 zero padding (inference fallback)
            B, _, H, W = fused_features.shape
            zero_streak = torch.zeros(B, 16, H, W,
                                      device=fused_features.device,
                                      dtype=fused_features.dtype)
            fused_features = torch.cat([fused_features, zero_streak], dim=1)

        # === Hybrid Backbone ===
        denoised = self.hybrid_backbone(fused_features, noisy_ct_normalized)  # (B, 1, H, W)

        # Noise prediction = input - denoised (역산)
        noise_pred = noisy_ct_normalized - denoised

        # Auxiliary outputs
        aux_outputs = {
            'noise_pred': noise_pred,
            'hu_mask': hu_mask,
            'spatial_coherence': spatial_coh,
            'wavelet_denoised': wavelet_denoised,
            'water_mask': water_mask,
            'bone_mask': bone_mask,
        }

        return denoised, aux_outputs
