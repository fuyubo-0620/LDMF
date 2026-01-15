import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn.bricks import build_norm_layer
import functools
from model.diffusers.module_util import (
    NonLinearity,
    Upsample, Downsample,
    default_conv,
    ResBlock,
    Identity)


class Encode(nn.Module):
    def __init__(self, in_ch=1, ch=64, ch_mult=[1, 2, 2, 4], embed_dim=8):
        super().__init__()
        self.depth = len(ch_mult)
        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())


        self.init_conv1 = default_conv(in_ch, ch, 3)
        self.init_conv2 = default_conv(in_ch, ch, 3)


        self.encoder1 = nn.ModuleList([])
        self.encoder2 = nn.ModuleList([])
        ch_mult_ext = [1] + ch_mult

        for i in range(self.depth):
            dim_in = ch * ch_mult_ext[i]
            dim_out = ch * ch_mult_ext[i + 1]


            self.encoder1.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in),
                Identity(),
                Downsample(dim_in, dim_out) if i != self.depth - 1 else default_conv(dim_in, dim_out)
            ]))
            self.encoder2.append(nn.ModuleList([
                block_class(dim_in=dim_in, dim_out=dim_in),
                Identity(),
                Downsample(dim_in, dim_out) if i != self.depth - 1 else default_conv(dim_in, dim_out)
            ]))


        mid_dim = ch * ch_mult_ext[-1]
        self.latent_conv1 = block_class(dim_in=mid_dim, dim_out=embed_dim)
        self.latent_conv2 = block_class(dim_in=mid_dim, dim_out=embed_dim)


        self.num_channels = [ch * ch_mult_ext[0]]
        for i in range(self.depth):
            self.num_channels.append(ch * ch_mult_ext[i])
        self.conv_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * c, c, kernel_size=3, padding=1),
                build_norm_layer(dict(type='BN'), c)[1],
                nn.LeakyReLU(0.2, True)
            ) for c in self.num_channels
        ])

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x1, x2):
        self.H, self.W = x1.shape[2:]
        x1 = self.check_image_size(x1, self.H, self.W)
        x2 = self.check_image_size(x2, self.H, self.W)

        x1 = self.init_conv1(x1)
        x2 = self.init_conv2(x2)


        h1 = [x1]
        h2 = [x2]
        for i, (b1, _, downsample) in enumerate(self.encoder1):
            x1 = b1(x1)

            h1.append(x1)
            x1 = downsample(x1)

        x1 = self.latent_conv1(x1)


        for i, (b1, _, downsample) in enumerate(self.encoder2):
            x2 = b1(x2)

            h2.append(x2)
            x2 = downsample(x2)
        x2 = self.latent_conv2(x2)


        h = []
        for i in range(len(h1)):
            fused = torch.cat([h1[i], h2[i]], dim=1)
            fused = self.conv_fuse[i](fused)
            h.append(fused)

        return x1, x2, h


class Decode(nn.Module):
    def __init__(self, out_ch=1, ch=64, ch_mult=[1, 2, 2, 4], embed_dim=8):
        super().__init__()
        self.depth = len(ch_mult)
        block_class = functools.partial(ResBlock, conv=default_conv, act=NonLinearity())


        self.decoder = nn.ModuleList([])
        ch_mult_ext = [1] + ch_mult

        for i in range(self.depth):
            dim_in = ch * ch_mult_ext[i]
            dim_out = ch * ch_mult_ext[i + 1]


            self.decoder.insert(0, nn.ModuleList([
                block_class(dim_in=dim_out + dim_in, dim_out=dim_out),  # 单层ResBlock
                Identity(),
                Upsample(dim_out, dim_in) if i != 0 else default_conv(dim_out, dim_in)
            ]))


        mid_dim = ch * ch_mult_ext[-1]
        self.post_latent_conv = block_class(dim_in=2 * embed_dim, dim_out=mid_dim)


        self.final_conv = nn.Conv2d(ch, out_ch, 3, 1, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

    def forward(self, x, h):

        x = self.post_latent_conv(x)


        for i, (b1, _, upsample) in enumerate(self.decoder):
            skip_feat = h[-(i + 1)]
            x = torch.cat([x, skip_feat], dim=1)
            x = b1(x)

            x = upsample(x)


        x = self.final_conv(x + h[0])
        x = torch.tanh(x)
        return x