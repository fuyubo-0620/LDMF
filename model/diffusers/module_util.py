import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch import einsum

from einops import rearrange, reduce


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


# Sinusoidal positional embeddings
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ Following @crowsonkb 's approach with random (optionally learned) sinusoidal positional embeddings """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def NonLinearity(inplace=False):
    return nn.SiLU(inplace)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# def Upsample(dim, dim_out=None):
#     return nn.Sequential(
#         nn.Upsample(scale_factor=2, mode='nearest'),
#         nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1)
#     )

def Upsample(dim, dim_out=None):
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Converse2D(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            scale=2,  # Upsampling ratio is 2x
            padding=2,
            padding_mode='circular',
            eps=1e-5
        ),
        nn.Conv2d(dim, dim_out, 3, 1, 1)  # Keep the original convolution layer for channel adjustment
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


def default_conv(dim_in, dim_out, kernel_size=3, bias=False):
    return nn.Conv2d(dim_in, dim_out, kernel_size, padding=(kernel_size // 2), bias=bias)


class Block(nn.Module):
    def __init__(self, conv, dim_in, dim_out, act=NonLinearity()):
        super().__init__()
        self.proj = conv(dim_in, dim_out)
        self.act = act

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, conv, dim_in, dim_out, time_emb_dim=None, act=NonLinearity()):
        super(ResBlock, self).__init__()
        self.mlp = nn.Sequential(
            act, nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim else None

        self.block1 = Block(conv, dim_in, dim_out, act)
        self.block2 = Block(conv, dim_out, dim_out, act)
        self.res_conv = conv(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)


# Channel attention
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


# Self-attention on each channel
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale a power of 2?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def initialize_weights(net_l, scale=1.):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale  # For residual blocks
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Converse2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=1, padding=2, padding_mode='circular', eps=1e-5):
        super(Converse2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        # Ensure depth-wise separable convolution
        assert self.out_channels == self.in_channels
        # Initialize weight parameters
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        # Initialize bias parameters
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        # Normalize weights using softmax
        self.weight.data = nn.functional.softmax(self.weight.data.view(1, self.in_channels, -1), dim=-1).view(1,
                                                                                                              self.in_channels,
                                                                                                              self.kernel_size,
                                                                                                              self.kernel_size)

    def forward(self, x):
        # Apply padding
        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding],
                                  mode=self.padding_mode, value=0)

        # Calculate bias with epsilon
        self.biaseps = torch.sigmoid(self.bias - 9.0) + self.eps
        _, _, h, w = x.shape
        # Upsample input
        STy = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            # Upsample using nearest neighbor interpolation
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
            # Bilinear interpolation can also be used
            # x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        # Convert weights to frequency domain representation
        FB = self.p2o(self.weight, (h * self.scale, w * self.scale))
        # Calculate conjugate
        FBC = torch.conj(FB)
        # Calculate squared magnitude
        F2B = torch.pow(torch.abs(FB), 2)
        # Calculate frequency domain product
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))

        # Frequency domain computation
        FR = FBFy + torch.fft.fftn(self.biaseps * x, dim=(-2, -1))
        x1 = FB.mul(FR)
        # Split and average
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        # Division operation
        invWBR = FBR.div(invW + self.biaseps)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR - FCBinvWBR) / self.biaseps
        # Inverse Fourier transform back to spatial domain
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        # Remove padding
        if self.padding > 0:
            out = out[..., self.padding * self.scale:-self.padding * self.scale,
                  self.padding * self.scale:-self.padding * self.scale]

        return out

    def splits(self, a, scale):
        '''
        Split tensor `a` into `scale x scale` distinct blocks.

        Args:
            a: Tensor of shape (..., W, H)
            scale: Split factor

        Returns:
            b: Tensor of shape (..., W/scale, H/scale, scale^2)
        '''
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape tensor to separate scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3,
                                                          len(leading_dims), len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()

        # Merge scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    def p2o(self, psf, shape):
        '''
        Convert point-spread function to optical transfer function.

        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.

        Args:
            psf: NxCxhxw
            shape: [H, W]

        Returns:
            otf: NxCxHxWx2
        '''
        # Initialize OTF tensor
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        # Copy PSF to OTF
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        # Roll to center
        otf = torch.roll(otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1))
        # Compute FFT
        otf = torch.fft.fftn(otf, dim=(-2, -1))

        return otf

    def upsample(self, x, scale=3):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros

        x: tensor image, NxCxWxH
        '''
        st = 0
        # Create zero-padded tensor
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)).type_as(x)
        # Fill original values at intervals of scale
        z[..., st::scale, st::scale].copy_(x)
        return z