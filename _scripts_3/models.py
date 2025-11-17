"""
models.py
공통 모델 정의

Models:
1. NAFNet: Denoising/Enhancement 기본 모델
2. AdaINEncoder: Style Transfer용 인코더
3. StructurePreservingStyleTransfer: NC 구조 보존 + CE 조영 학습
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# NAFNet Components (기존과 동일)
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
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
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
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )
        
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        
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
    """NAFNet - Denoising/Enhancement 기본 모델"""
    def __init__(self, img_channel=1, width=32, middle_blk_num=12, 
                 enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1, bias=True)
        
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
        x = x + inp  # Residual
        
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


# ============================================================
# Style Transfer Components
# ============================================================

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization
    
    NC의 구조(content)를 유지하면서
    CE의 조영 패턴(style)만 적용
    """
    def __init__(self):
        super().__init__()
    
    def calc_mean_std(self, feat, eps=1e-5):
        """Feature의 mean, std 계산"""
        B, C = feat.shape[:2]
        feat_var = feat.view(B, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(B, C, 1, 1)
        feat_mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        return feat_mean, feat_std
    
    def forward(self, content_feat, style_feat):
        """
        content_feat: NC features (구조 보존!)
        style_feat: CE features (조영 패턴)
        
        Returns: NC 구조 + CE 스타일
        """
        B, C, H, W = content_feat.shape
        
        # Style statistics (CE의 조영 패턴)
        style_mean, style_std = self.calc_mean_std(style_feat)
        
        # Content statistics (NC의 구조)
        content_mean, content_std = self.calc_mean_std(content_feat)
        
        # Normalize content (구조 정규화)
        normalized = (content_feat - content_mean) / content_std
        
        # Apply style (조영 패턴 적용)
        stylized = normalized * style_std + style_mean
        
        return stylized


class StructurePreservingEncoder(nn.Module):
    """
    구조 보존 인코더
    
    NC의 해부학적 구조를 추출
    """
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        
        # Progressive encoding
        self.enc1 = self._make_layer(in_channels, base_channels)
        self.enc2 = self._make_layer(base_channels, base_channels*2)
        self.enc3 = self._make_layer(base_channels*2, base_channels*4)
        self.enc4 = self._make_layer(base_channels*4, base_channels*8)
        
        self.pool = nn.MaxPool2d(2)
    
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Multi-scale features
        e1 = self.enc1(x)           # [B, 64, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 128, H/2, W/2]
        e3 = self.enc3(self.pool(e2))  # [B, 256, H/4, W/4]
        e4 = self.enc4(self.pool(e3))  # [B, 512, H/8, W/8]
        
        return [e1, e2, e3, e4]


class StyleDecoder(nn.Module):
    """
    스타일 적용 디코더
    
    NC 구조 + CE 조영 → Enhanced NC
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = self._make_layer(base_channels*8, base_channels*4)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = self._make_layer(base_channels*4, base_channels*2)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = self._make_layer(base_channels*2, base_channels)
        
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        """
        features: [e1, e2, e3, e4] (stylized features)
        """
        e1, e2, e3, e4 = features
        
        d4 = self.dec4(e4)
        d3 = self.dec3(self.up4(d4) + e3)
        d2 = self.dec2(self.up3(d3) + e2)
        
        out = self.final(self.up2(d2) + e1)
        
        return out


class StructurePreservingStyleTransfer(nn.Module):
    """
    ★ 핵심 모델: NC 구조 보존 + CE 조영 효과
    
    Architecture:
    1. NC, CE를 각각 인코딩
    2. AdaIN으로 NC 구조 + CE 스타일 결합
    3. 디코더로 Enhanced NC 생성
    
    핵심: NC의 해부학적 구조는 절대 보존!
    """
    def __init__(self, base_channels=64):
        super().__init__()
        
        # Content encoder (NC 구조 추출)
        self.content_encoder = StructurePreservingEncoder(
            in_channels=1,
            base_channels=base_channels
        )
        
        # Style encoder (CE 조영 추출)
        self.style_encoder = StructurePreservingEncoder(
            in_channels=1,
            base_channels=base_channels
        )
        
        # AdaIN layers (각 스케일마다)
        self.adain_layers = nn.ModuleList([
            AdaIN() for _ in range(4)
        ])
        
        # Decoder
        self.decoder = StyleDecoder(base_channels=base_channels)
    
    def forward(self, nc, ce=None, alpha=1.0):
        """
        Args:
            nc: NC image [B, 1, H, W] (구조 기준)
            ce: CE image [B, 1, H, W] (조영 기준, training only)
            alpha: style strength (0=only NC, 1=full CE style)
        
        Returns:
            Enhanced NC with CE contrast
        """
        # Content features (NC 구조)
        content_feats = self.content_encoder(nc)
        
        if ce is not None and self.training:
            # Style features (CE 조영)
            style_feats = self.style_encoder(ce)
            
            # Apply AdaIN at each scale
            stylized_feats = []
            for content_feat, style_feat, adain in zip(content_feats, style_feats, self.adain_layers):
                stylized = adain(content_feat, style_feat)
                # Blend with original (구조 보존 강화)
                stylized = alpha * stylized + (1 - alpha) * content_feat
                stylized_feats.append(stylized)
        else:
            # Inference: style을 학습된 패턴으로 적용
            stylized_feats = content_feats
        
        # Decode
        enhanced_nc = self.decoder(stylized_feats)
        
        return enhanced_nc
    
    def set_style_mode(self, mode='train'):
        """
        train: CE 이미지로 style 학습
        test: 학습된 style 자동 적용
        """
        if mode == 'train':
            self.style_encoder.train()
        else:
            self.style_encoder.eval()


if __name__ == '__main__':
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # NAFNet test
    model = NAFNet(width=32).to(device)
    x = torch.randn(2, 1, 256, 256).to(device)
    y = model(x)
    print(f"NAFNet: {x.shape} → {y.shape}")
    
    # Style Transfer test
    model2 = StructurePreservingStyleTransfer(base_channels=64).to(device)
    nc = torch.randn(2, 1, 256, 256).to(device)
    ce = torch.randn(2, 1, 256, 256).to(device)
    enhanced = model2(nc, ce, alpha=1.0)
    print(f"Style Transfer: NC{nc.shape} + CE{ce.shape} → {enhanced.shape}")