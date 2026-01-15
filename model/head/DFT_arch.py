from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from model.head.mamba_simple import mamba2
from model.mamba_simple import SS2D


def _feat2seq(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, C, H, W = x.shape
    seq = rearrange(x, 'b c h w -> b (h w) c')
    return seq, (H, W)


def _seq2feat(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    H, W = size
    feat = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
    return feat


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x * torch.sigmoid(x)


class DWT(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        assert wavelet == 'haar'

        haar_row = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32) * (1.0 / math.sqrt(2))
        haar_col = haar_row.T

        self.register_buffer('row_kernel', haar_row.unsqueeze(0).unsqueeze(0))
        self.register_buffer('col_kernel', haar_col.unsqueeze(0).unsqueeze(0))

        row_kernel = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32).view(2, 1, 1, 2) * (
                    1.0 / math.sqrt(2))
        col_kernel = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32).view(2, 1, 2, 1) * (
                    1.0 / math.sqrt(2))

        self.register_buffer('row_conv_kernel', row_kernel)
        self.register_buffer('col_conv_kernel', col_kernel)

    def forward(self, x):
        B, C, H, W = x.shape

        pad_h = 1 if H % 2 != 0 else 0
        pad_w = 1 if W % 2 != 0 else 0
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = x.shape[-2], x.shape[-1]

        x_row = F.conv2d(x, self.row_conv_kernel.repeat(C, 1, 1, 1), stride=(1, 2), groups=C)
        x_low_row = x_row[:, ::2, :, :]
        x_high_row = x_row[:, 1::2, :, :]

        ll = F.conv2d(x_low_row, self.col_conv_kernel.repeat(C, 1, 1, 1), stride=(2, 1), groups=C)[:, ::2, :, :]
        hl = F.conv2d(x_low_row, self.col_conv_kernel.repeat(C, 1, 1, 1), stride=(2, 1), groups=C)[:, 1::2, :, :]
        lh = F.conv2d(x_high_row, self.col_conv_kernel.repeat(C, 1, 1, 1), stride=(2, 1), groups=C)[:, ::2, :, :]
        hh = F.conv2d(x_high_row, self.col_conv_kernel.repeat(C, 1, 1, 1), stride=(2, 1), groups=C)[:, 1::2, :, :]

        return ll, lh, hl, hh


class IDWT(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        assert wavelet == 'haar'

        row_kernel = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32).view(2, 1, 1, 2) * (
                    1.0 / math.sqrt(2))
        col_kernel = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32).view(2, 1, 2, 1) * (
                    1.0 / math.sqrt(2))

        self.register_buffer('row_deconv_kernel', row_kernel)
        self.register_buffer('col_deconv_kernel', col_kernel)

    def forward(self, coeffs):
        ll, lh, hl, hh = coeffs
        B, C, H, W = ll.shape

        assert lh.shape == (B, C, H, W) and hl.shape == (B, C, H, W) and hh.shape == (B, C, H, W)

        col_low = torch.cat([ll, hl], dim=1)
        col_high = torch.cat([lh, hh], dim=1)

        col_low = F.conv_transpose2d(col_low, self.col_deconv_kernel.repeat(C, 1, 1, 1), stride=(2, 1), groups=C)
        col_high = F.conv_transpose2d(col_high, self.col_deconv_kernel.repeat(C, 1, 1, 1), stride=(2, 1), groups=C)

        row_cat = torch.cat([col_low, col_high], dim=1)
        x = F.conv_transpose2d(row_cat, self.row_deconv_kernel.repeat(C, 1, 1, 1), stride=(1, 2), groups=C)

        return x


class MultiKernelConv2d(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1, 3, 5]):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes

        num_kernels = len(kernel_sizes)
        base_channels = in_channels // num_kernels
        remainder = in_channels % num_kernels
        self.out_channels_per_kernel = [base_channels + 1 if i < remainder else base_channels for i in
                                        range(num_kernels)]

        self.conv_layers = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            out_ch = self.out_channels_per_kernel[i]
            padding = k // 2
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_ch, kernel_size=k, padding=padding)
            )

        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.dilated_conv = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)

        self.norm = nn.GroupNorm(1, in_channels)
        self.activation = nn.SiLU()

    def forward(self, x):
        branch_outs = [conv(x) for conv in self.conv_layers]
        x_concat = torch.cat(branch_outs, dim=1)

        x_fused = self.fusion_conv(x_concat)
        x_dw = self.pointwise_conv(self.depthwise_conv(x))
        x_dilated = self.dilated_conv(x)

        x_out = x + x_fused + x_dw + x_dilated
        x_out = self.activation(self.norm(x_out))

        return x_out


class DTB(nn.Module):
    def __init__(self, dim, scale=1, time_embed_dim=128, conv_kernel=4):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.time_embed_dim = time_embed_dim

        assert dim % 16 == 0
        self.branch_dim = dim // 2

        self.time_to_channel = nn.Linear(time_embed_dim, self.branch_dim)

        self.norm_branch1 = nn.GroupNorm(1, self.branch_dim)
        self.norm_branch2 = nn.GroupNorm(1, self.branch_dim)

        self.dwt = DWT()
        self.idwt = IDWT()

        self.lh_ss2d = SS2D(d_model=self.branch_dim, scan_type='lh')
        self.hl_ss2d = SS2D(d_model=self.branch_dim, scan_type='hl')
        self.hh_ss2d = SS2D(d_model=self.branch_dim, scan_type='hh')

        self.multiconv = MultiKernelConv2d(in_channels=self.branch_dim)

        self.x1_scaler = self._build_scaler(conv_kernel, scale)
        self.ll_conv = nn.Conv2d(self.branch_dim, self.branch_dim, 1, 1, 0, bias=False)

        self.seq_norm = nn.LayerNorm(self.branch_dim)

        self.mamba = mamba2(d_model=self.branch_dim)

        self.fusion_norm = nn.GroupNorm(1, dim)

        self.x1_restorer = self._build_restorer(conv_kernel, scale)

    def _build_scaler(self, kernel_size, scale):
        if scale == 0.5:
            return nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size, stride=2, padding=kernel_size // 2,
                             bias=False)
        elif scale == 2:
            return nn.ConvTranspose2d(self.branch_dim, self.branch_dim, kernel_size, stride=2, padding=kernel_size // 2,
                                      output_padding=1, bias=False)
        else:
            return nn.Identity()

    def _build_restorer(self, kernel_size, scale):
        if scale == 0.5:
            return nn.ConvTranspose2d(self.branch_dim, self.branch_dim, kernel_size, stride=2, padding=kernel_size // 2,
                                      output_padding=1, bias=False)
        elif scale == 2:
            return nn.Conv2d(self.branch_dim, self.branch_dim, kernel_size, stride=2, padding=kernel_size // 2,
                             bias=False)
        else:
            return nn.Identity()

    def _pad_seq_to_8x(self, seq: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, L, C = seq.shape
        pad_len = (8 - L % 8) % 8
        if pad_len > 0:
            seq = F.pad(seq, (0, 0, 0, pad_len))
        return seq, pad_len

    def _process_low_freq(self, ll: torch.Tensor) -> torch.Tensor:
        ll = self.ll_conv(ll)
        ll_seq, (H, W) = _feat2seq(ll)
        ll_seq = ll_seq.contiguous()
        ll_seq, pad_len = self._pad_seq_to_8x(ll_seq)
        ll_seq = self.seq_norm(ll_seq)
        ll_seq, _ = self.mamba(ll_seq)
        if pad_len > 0:
            ll_seq = ll_seq[:, :-pad_len, :]
        ll_feat = _seq2feat(ll_seq, (H, W))
        return ll_feat

    def _process_high_freq(self, hf: torch.Tensor, ss2d_module: nn.Module) -> torch.Tensor:
        hf = hf.permute(0, 2, 3, 1).contiguous()
        hf = ss2d_module(hf)
        hf = hf.permute(0, 3, 1, 2).contiguous()
        return hf

    def forward(self, x_feat: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x_feat.shape
        assert C == self.dim

        x1 = x_feat[:, :self.branch_dim, :, :]
        x2 = x_feat[:, self.branch_dim:, :, :]

        if len(t.size()) == 1:
            t_emb = get_timestep_embedding(t, self.time_embed_dim)
            t = nonlinearity(t_emb)
        t_branch = self.time_to_channel(t).unsqueeze(-1).unsqueeze(-1)

        x1 = self.norm_branch1(x1) + t_branch
        x1_scaled = self.x1_scaler(x1)
        ll, lh, hl, hh = self.dwt(x1_scaled)

        ll = self._process_low_freq(ll)
        lh = self._process_high_freq(lh, self.lh_ss2d)
        hl = self._process_high_freq(hl, self.hl_ss2d)
        hh = self._process_high_freq(hh, self.hh_ss2d)

        target_size = ll.shape[2:]
        if lh.shape[2:] != target_size:
            lh = F.interpolate(lh, size=target_size, mode='bilinear', align_corners=False)
        if hl.shape[2:] != target_size:
            hl = F.interpolate(hl, size=target_size, mode='bilinear', align_corners=False)
        if hh.shape[2:] != target_size:
            hh = F.interpolate(hh, size=target_size, mode='bilinear', align_corners=False)

        x1_recon = self.idwt((ll, lh, hl, hh))
        x1_recon = self.x1_restorer(x1_recon)
        x1_recon = x1_recon[..., :x1.shape[2], :x1.shape[3]]
        x1 = x1 + x1_recon

        x2 = self.norm_branch2(x2) + t_branch
        x2 = self.multiconv(x2)

        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.fusion_norm(x_fused)

        return x_fused, t


class DFT(nn.Module):
    def __init__(self,
                 inp_channels=16,
                 out_channels=16,
                 base_dim=128,
                 img_size=32):
        super().__init__()
        self.img_size = img_size
        self.base_dim = base_dim

        assert base_dim % 16 == 0
        self.init_proj = nn.Conv2d(inp_channels, base_dim, 3, 1, 1, bias=False)

        self.enc2 = DTB(dim=base_dim, scale=0.5)
        self.enc3 = DTB(dim=base_dim, scale=0.5)

        self.latent = DTB(dim=base_dim, scale=1)

        self.dec1 = DTB(dim=base_dim, scale=2)
        self.dec2 = DTB(dim=base_dim, scale=2)

        self.final_out = nn.Conv2d(base_dim, out_channels, 1, 1, 0)

        self.reduce_bottleneck = nn.Conv2d(base_dim, 16, kernel_size=1, bias=False)
        self.reduce_up1 = nn.Conv2d(base_dim, 16, kernel_size=1, bias=False)
        self.reduce_up2 = nn.Conv2d(base_dim, 16, kernel_size=1, bias=False)

    def forward(self, x, t, device):
        if isinstance(t, int) or isinstance(t, float) or t.dim() == 0:
            t = torch.tensor([t], device=device, dtype=torch.float32)
        elif len(t.shape) == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0])

        x = self.init_proj(x)
        B, C, H, W = x.shape

        x, t = self.enc2(x, t)
        x, t = self.enc3(x, t)

        x, t = self.latent(x, t)
        feat_bottleneck = self.reduce_bottleneck(x)

        x, t = self.dec1(x, t)
        feat_up1 = self.reduce_up1(x)
        x, t = self.dec2(x, t)
        feat_up2 = self.reduce_up2(x)

        x_out = self.final_out(x)
        x_out = x_out[..., :H, :W]

        middle_feat = [feat_bottleneck, feat_up1, feat_up2]
        return x_out, middle_feat


if __name__ == "__main__":
    model = DFT(inp_channels=16, out_channels=16, base_dim=128, img_size=64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    x = torch.randn(16, 16, 32, 32).to(device)
    t = torch.randint(0, 1000, (16,)).to(device)

    for _ in range(5):
        with torch.no_grad():
            out, feats = model(x, t, device)

    import time

    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            out, feats = model(x, t, device)
    end = time.time()

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"中间特征数量: {len(feats)}")
    print(f"瓶颈块特征形状: {feats[0].shape}")
    print(f"第一次上采样特征形状: {feats[1].shape}")
    print(f"第二次上采样特征形状: {feats[2].shape}")
    print(f"100次运行耗时: {end - start:.4f}秒")