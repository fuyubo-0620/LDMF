import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
from einops import rearrange

class CrossMambaFusion(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, in_channels=None, out_channels=None, if_devide_out=False):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, d_model, 1)
        self.decoder = nn.Conv2d(d_model, out_channels, 1)
    def forward(self, vis, inf):
        x = self.encoder(vis) + self.encoder(inf)
        return self.decoder(x)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x

class channelInteraction(nn.Module):
    def __init__(self, channelin, channelout=None):
        super(channelInteraction, self).__init__()
        if channelout is None:
            channelout = channelin

        self.split_channels = channelin // 2

        self.chaAtten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channelin, max(4, channelin // 8), kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channelin // 8), channelin, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.reflashChaAtten1 = nn.Sequential(
            nn.Conv2d(channelin, max(4, channelin // 8), kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channelin // 8), channelin, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.reflashFused1 = nn.Sequential(
            nn.Conv2d(channelin, channelin, 3, 1, 1, groups=channelin),
            nn.Conv2d(channelin, channelin, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channelin, channelin, 3, 1, 1, groups=channelin),
            nn.Conv2d(channelin, channelin, 1, 1, 0)
        )

        self.postprocess = nn.Sequential(
            nn.Conv2d(channelin, channelin, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        if channelin != channelout:
            self.final_conv = nn.Conv2d(channelin, channelout, 1, 1, 0)
        else:
            self.final_conv = nn.Identity()

    def forward(self, x):
        vis = x[:, :self.split_channels, :, :]
        inf = x[:, self.split_channels:, :, :]
        vis_cat = torch.cat([vis, inf], 1)

        chanAtten = self.chaAtten(vis_cat)
        fused_OneOrderCha = vis_cat * chanAtten

        fused_OneOrderCha = self.reflashFused1(fused_OneOrderCha)
        chanAttenReflash1 = self.reflashChaAtten1(chanAtten)
        fused_twoOrderCha = fused_OneOrderCha * chanAttenReflash1

        fused_twoOrderCha = self.postprocess(fused_twoOrderCha)
        fused_twoOrderCha = self.final_conv(fused_twoOrderCha)

        if fused_twoOrderCha.shape[1] == x.shape[1]:
            fused = fused_twoOrderCha + x
        else:
            residual_conv = nn.Conv2d(x.shape[1], fused_twoOrderCha.shape[1], 1, 1, 0).to(x.device)
            fused = fused_twoOrderCha + residual_conv(x)

        return fused

class spatialInteraction(nn.Module):
    def __init__(self, channelin, channelout=None):
        super(spatialInteraction, self).__init__()

        if channelout is None:
            channelout = channelin

        self.split_channels = channelin // 2

        self.reflashFused = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.split_channels, self.split_channels, 3, 1, 1)
        )

        self.reflashInfrared = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.split_channels, self.split_channels, 3, 1, 1)
        )

        self.norm = LayerNorm(self.split_channels, LayerNorm_type='WithBias')

        self.final_conv = nn.Conv2d(self.split_channels, channelout, 1, 1, 0)

    def forward(self, x):
        vis = x[:, :self.split_channels, :, :]
        inf = x[:, self.split_channels:, :, :]

        _, C, H, W = vis.size()

        vis_fft = torch.fft.rfft2(vis, norm='ortho')
        inf_fft = torch.fft.rfft2(inf, norm='ortho')

        atten = vis_fft * inf_fft
        atten = torch.fft.irfft2(atten, s=(H, W), norm='ortho')
        atten = self.norm(atten)
        fused_OneOrderSpa = atten * inf

        fused_OneOrderSpa = self.reflashFused(fused_OneOrderSpa)
        infraredReflash = self.reflashInfrared(inf)
        fused_twoOrderSpa = fused_OneOrderSpa * infraredReflash

        fused_output = self.final_conv(fused_twoOrderSpa)

        if fused_output.shape[1] == vis.shape[1]:
            fused = fused_output + vis
        else:
            residual_conv = nn.Conv2d(vis.shape[1], fused_output.shape[1], 1, 1, 0).to(x.device)
            fused = fused_output + residual_conv(vis)

        return fused

class crossMambaHead(nn.Module):
    def __init__(self, in_ch, d_model, img_size):
        super(crossMambaHead, self).__init__()

        self.split_channels = in_ch // 2
        self.d_model = d_model
        self.img_size = img_size

        self.cr_mamba = CrossMambaFusion(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            in_channels=self.split_channels,
            out_channels=16,
            if_devide_out=False
        )

        self.cr_mamba2 = CrossMambaFusion(
            d_model=d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            in_channels=self.split_channels,
            out_channels=16,
            if_devide_out=False
        )

    def forward(self, x):
        vis = x[:, :self.split_channels, :, :]
        inf = x[:, self.split_channels:, :, :]

        feat1 = self.cr_mamba(vis, inf)
        feat2 = self.cr_mamba2(inf, vis)

        fuse_feat = torch.cat((feat1, feat2), dim=1)

        return fuse_feat

class HeadTanh2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(HeadTanh2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
                              dilation=dilation, groups=groups)

    def forward(self, x):
        return torch.tanh(self.conv(x)) / 2 + 0.5

class Fusion_head(nn.Module):
    def __init__(self,
                 out_channels=16,
                 dim=256,
                 bias=False,
                 img_size=(32, 32),
                 feat_channels=[16, 16, 16]
                 ):
        super(Fusion_head, self).__init__()

        self.channel_adjust_blocks = nn.ModuleList()
        for c in feat_channels:
            self.channel_adjust_blocks.append(
                nn.Conv2d(c, c, 3, padding=1, bias=bias)
            )

        self.feat_fusion = nn.Sequential(
            nn.Conv2d(sum(feat_channels), feat_channels[-1], 1, bias=bias),
            nn.ReLU(inplace=True)
        )

        self.cross_mamba_head = crossMambaHead(
            in_ch=feat_channels[-1],
            d_model=dim,
            img_size=img_size
        )

        self.latent_adjust = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=1,
            bias=bias
        )

        self.final_fusion = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1, bias=bias)
        )

        self.proj = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1)

    def forward(self, feat, latent_result):
        feat_small, feat_medium, feat_large = feat

        x_small = self.channel_adjust_blocks[0](feat_small)
        x_medium = self.channel_adjust_blocks[1](feat_medium)
        x_large = self.channel_adjust_blocks[2](feat_large)

        x_fused = x_small + x_medium + x_large

        mamba_fused = self.cross_mamba_head(x_fused)

        latent_result = self.latent_adjust(latent_result)
        x_final2 = mamba_fused + latent_result
        x = self.final_fusion(x_final2)

        return x

if __name__ == "__main__":
    model = Fusion_head(
        feat_channels=[16, 16, 16],
        img_size=(32, 32)
    )

    feat_small = torch.randn(16, 16, 32, 32)
    feat_medium = torch.randn(16, 16, 32, 32)
    feat_large = torch.randn(16, 16, 32, 32)
    feat = (feat_small, feat_medium, feat_large)
    latent_result = torch.randn(16, 16, 32, 32)

    output = model(feat, latent_result)
    print(f"输出形状：{output.shape}")