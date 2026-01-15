import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    print("✓ 成功导入 causal_conv1d_fn")
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
    print("⚠ 无法导入 causal_conv1d 优化函数，将使用标准卷积路径")

selective_scan_fn = None
mamba_inner_fn = None
bimamba_inner_fn = None
mamba_inner_fn_no_out_proj = None

import_paths = [
    ('mamba_ssm.ops.triton.selective_scan', 'selective_scan_fn'),
    ('mamba_ssm.ops.selective_scan_interface', 'selective_scan_fn'),
    ('mamba_ssm.ops.triton.ssd_combined', 'mamba_chunk_scan_combined'),
]

for module_path, func_name in import_paths:
    try:
        if func_name == 'mamba_chunk_scan_combined':
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
            selective_scan_fn = mamba_chunk_scan_combined
            print(f"✓ 成功导入 '{func_name}' (Mamba-2 分块扫描)")
            break
        else:
            import importlib
            module = importlib.import_module(module_path)
            selective_scan_fn = getattr(module, func_name)
            print(f"✓ 成功导入 '{func_name}' from '{module_path}'")
            if module_path == 'mamba_ssm.ops.selective_scan_interface':
                try:
                    mamba_inner_fn = getattr(module, 'mamba_inner_fn')
                    bimamba_inner_fn = getattr(module, 'bimamba_inner_fn')
                    mamba_inner_fn_no_out_proj = getattr(module, 'mamba_inner_fn_no_out_proj')
                except AttributeError:
                    pass
            break
    except (ImportError, AttributeError) as e:
        print(f"❌ 从 '{module_path}' 导入 '{func_name}' 失败: {e}")
        continue

if selective_scan_fn is None:
    print("⚠ 所有标准导入路径都失败，使用简化回退函数")
    def simple_selective_scan(x, dt, A, B, C, D, z=None, delta_bias=None, delta_softplus=True, return_last_state=False):
        print("⚠ 警告：使用简化回退函数，SSM功能受限")
        batch, dim, seqlen = x.shape
        if return_last_state:
            return torch.zeros_like(x), torch.zeros(batch, dim, A.shape[-1], device=x.device)
        return torch.zeros_like(x)
    selective_scan_fn = simple_selective_scan

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
    print("✓ 成功导入 RMSNormGated")
except ImportError:
    print("⚠ 无法导入 RMSNormGated，将使用自定义实现")
    class RMSNormGated(nn.Module):
        def __init__(self, dim, eps=1e-5, norm_before_gate=False, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps
            self.norm_before_gate = norm_before_gate
        def forward(self, x, z=None):
            if self.norm_before_gate:
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                x_norm = x * rms
                x_out = x_norm * self.weight
                if z is not None:
                    x_out = x_out * F.silu(z)
            else:
                if z is not None:
                    x = x * F.silu(z)
                rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                x_out = x * rms * self.weight
            return x_out

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    print("✓ 成功导入 selective_state_update")
except ImportError:
    selective_state_update = None
    print("⚠ 未导入 selective_state_update")

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
    print("✓ 成功导入 RMSNorm, layer_norm_fn, rms_norm_fn")
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    print("⚠ 未导入优化的 LayerNorm 函数")

print("-" * 60)

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            scan_type='hl',
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.scantype = scan_type
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj_hl = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_hl_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj_hl], dim=0))
        del self.x_proj_hl
        self.dt_hl_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,** factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_hl_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_hl_projs], dim=0))
        self.dt_projs_hl_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_hl_projs], dim=0))
        del self.dt_hl_projs
        self.A_logs_hl = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds_hl = self.D_init(self.d_inner, copies=2, merge=True)

        self.x_proj_lh = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_lh_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj_lh], dim=0))
        del self.x_proj_lh
        self.dt_lh_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_lh_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_lh_projs], dim=0))
        self.dt_projs_lh_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_lh_projs], dim=0))
        del self.dt_lh_projs
        self.A_logs_lh = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds_lh = self.D_init(self.d_inner, copies=2, merge=True)

        self.x_proj_hh = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_hh_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj_hh], dim=0))
        del self.x_proj_hh
        self.dt_hh_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_hh_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_hh_projs], dim=0))
        self.dt_projs_hh_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_hh_projs], dim=0))
        del self.dt_hh_projs
        self.A_logs_hh = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)
        self.Ds_hh = self.D_init(self.d_inner, copies=2, merge=True)

        self.forward_corehl = self.forward_corehl
        self.forward_corelh = self.forward_corelh
        self.forward_corehh = self.forward_corehh
        self.diagonal_trans = self.diagonal_trans
        self.reverse_diagonal_trans = self.reverse_diagonal_trans

        self.out_normhl = nn.LayerNorm(self.d_inner)
        self.out_projhl = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_normlh = nn.LayerNorm(self.d_inner)
        self.out_projlh = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_normhh = nn.LayerNorm(self.d_inner)
        self.out_projhh = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corehl(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 2
        x_inv = torch.flip(x, dims=[-1])
        xs = torch.stack([x.view(B, -1, L), x_inv.view(B, -1, L)], dim=1).view(B, 2, -1, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_hl_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_hl_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds_hl.float().view(-1)
        As = -torch.exp(self.A_logs_hl.float()).view(-1, self.d_state)
        dt_projs_hl_bias = self.dt_projs_hl_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_hl_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, 1], dims=[-1])
        return out_y[:, 0], inv_y

    def forward_corelh(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 2
        x_hwwh = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        x_inv_hwwh = torch.flip(x_hwwh, dims=[-1]).view(B, -1, L)
        xs = torch.stack([x_hwwh, x_inv_hwwh], dim=1).view(B, 2, -1, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_lh_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_lh_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds_lh.float().view(-1)
        As = -torch.exp(self.A_logs_lh.float()).view(-1, self.d_state)
        dt_projs_lh_bias = self.dt_projs_lh_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_lh_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        wh_y = torch.transpose(out_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, 1, -1, L)
        invwh_y = torch.transpose(inv_y[:, 0].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return wh_y, invwh_y

    def diagonal_trans(self, x: torch.Tensor, H, W):
        B, K, _, L = x.shape
        assert L == H * W, "最后一个维度L必须等于H*W"
        idx = torch.arange(H * W, device=x.device).reshape(H, W)
        i_idx = torch.arange(H, device=x.device).reshape(-1, 1).expand(H, W)
        j_idx = torch.arange(W, device=x.device).reshape(1, -1).expand(H, W)
        diag_mask = (i_idx + j_idx)
        sorted_idx = torch.argsort(diag_mask.view(-1))
        diag_indices = torch.index_select(idx.view(-1), 0, sorted_idx)
        x_flat = x.view(B, K, -1, H * W)
        diag_flat = torch.index_select(x_flat, dim=3, index=diag_indices)
        return diag_flat, diag_indices

    def reverse_diagonal_trans(self, x: torch.Tensor, diag_indices):
        B, K, _, L = x.shape
        x_flat = x.view(B, K, -1, L)
        reverse_indices = torch.argsort(diag_indices)
        original_flat = torch.index_select(x_flat, dim=3, index=reverse_indices)
        return original_flat

    def forward_corehh(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 2
        x_invhh = torch.flip(x, dims=[-1])
        xs = torch.stack([x.view(B, -1, L), x_invhh.view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs, diag_indices = self.diagonal_trans(xs, H, W)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_hh_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_hh_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds_hh.float().view(-1)
        As = -torch.exp(self.A_logs_hh.float()).view(-1, self.d_state)
        dt_projs_hh_bias = self.dt_projs_hh_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_hh_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        trans_y = self.reverse_diagonal_trans(out_y, diag_indices)
        inv_y = torch.flip(trans_y[:, 1], dims=[-1])
        return trans_y[:, 0], inv_y

    def forward(self, x: torch.Tensor, **kwargs):
        if self.scantype == 'hl':
            B, H, W, C = x.shape
            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
            yhl1, yhl2 = self.forward_corehl(x)
            assert yhl1.dtype == torch.float32
            yhl = yhl1 + yhl2
            yhl = torch.transpose(yhl, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            yhl = self.out_normhl(yhl)
            yhl = yhl * F.silu(z)
            outhl = self.out_projhl(yhl)
            if self.dropout is not None:
                outhl = self.dropout(outhl)
            return outhl

        if self.scantype == 'lh':
            B, H, W, C = x.shape
            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
            ylh1, ylh2 = self.forward_corelh(x)
            assert ylh1.dtype == torch.float32
            ylh = ylh1 + ylh2
            ylh = torch.transpose(ylh, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            ylh = self.out_normlh(ylh)
            ylh = ylh * F.silu(z)
            outlh = self.out_projlh(ylh)
            if self.dropout is not None:
                outlh = self.dropout(outlh)
            return outlh

        if self.scantype == 'hh':
            B, H, W, C = x.shape
            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))
            yhh1, yhh2 = self.forward_corehh(x)
            assert yhh1.dtype == torch.float32
            yhh = yhh1 + yhh2
            yhh = torch.transpose(yhh, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            yhh = self.out_normhh(yhh)
            yhh = yhh * F.silu(z)
            outhh = self.out_projhh(yhh)
            if self.dropout is not None:
                outhh = self.dropout(outhh)
            return outhh

class CrossMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            cross_mode='bi_cross',
            if_devide_out=False,
            in_channels=None,
            out_channels=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.cross_mode = cross_mode
        self.if_devide_out = if_devide_out
        self.device = device

        self.use_feat_map = (in_channels is not None) and (out_channels is not None)
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.use_feat_map:
            self.encoder = nn.Conv2d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=1,
                bias=bias,** factory_kwargs
            )
            self.decoder = nn.Conv2d(
                in_channels=d_model,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
                **factory_kwargs
            )

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,** factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank **-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        if cross_mode in ['bi_same', 'bi_cross', 'bi_swap']:
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,** factory_kwargs,
            )
            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            dt_b = torch.exp(
                torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt_b = dt_b + torch.log(-torch.expm1(-dt_b))
            with torch.no_grad():
                self.dt_proj_b.bias.copy_(inv_dt_b)
            self.dt_proj_b.bias._no_reinit = True

            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False)

    def _encode_feat(self, x: torch.Tensor) -> (torch.Tensor, int, int):
        B, C, H, W = x.shape
        x = self.encoder(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x, H, W

    def _decode_seq(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = x.shape[0]
        x = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        x = self.decoder(x)
        return x

    def forward(self,
                hidden_states_a: torch.Tensor,
                hidden_states_b: Optional[torch.Tensor] = None,
                inference_params=None) -> torch.Tensor:
        encode_flag = False
        H, W = 0, 0
        if self.use_feat_map and hidden_states_a.dim() == 4:
            encode_flag = True
            hidden_states_a, H, W = self._encode_feat(hidden_states_a)
            if hidden_states_b is not None and hidden_states_b.dim() == 4:
                hidden_states_b, _, _ = self._encode_feat(hidden_states_b)

        batch, seqlen, dim = hidden_states_a.shape
        if self.cross_mode in ['bi_cross', 'bi_swap'] and hidden_states_b is None:
            raise ValueError(f"模式 {self.cross_mode} 需要两个输入，但只提供了一个")

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states_a, conv_state, ssm_state)
                if encode_flag:
                    out = self._decode_seq(out, H, W)
                return out

        if self.cross_mode == 'single':
            out = self._forward_single(hidden_states_a, conv_state, ssm_state)
        elif self.cross_mode == 'bi_same':
            out = self._forward_bi_same(hidden_states_a, conv_state, ssm_state)
        elif self.cross_mode == 'bi_cross':
            out = self._forward_bi_cross(hidden_states_a, hidden_states_b, conv_state, ssm_state)
        elif self.cross_mode == 'bi_swap':
            out = self._forward_bi_swap(hidden_states_a, hidden_states_b, conv_state, ssm_state)
        else:
            raise ValueError(f"未知的模式: {self.cross_mode}")

        if encode_flag:
            out = self._decode_seq(out, H, W)

        return out

    def _forward_single(self, x: torch.Tensor, conv_state=None, ssm_state=None):
        batch, seqlen, dim = x.shape
        xz = self.in_proj(x)
        x_forward, z = xz.chunk(2, dim=-1)

        if conv_state is not None:
            conv_state.copy_(F.pad(x_forward.transpose(1, 2), (self.d_conv - x_forward.shape[1], 0)))

        if causal_conv1d_fn is not None and x_forward.is_cuda:
            weight = rearrange(self.conv1d.weight, "d 1 w -> d w").to(x_forward.device)
            bias = self.conv1d.bias.to(x_forward.device) if self.conv1d.bias is not None else None
            x_forward = causal_conv1d_fn(
                x=x_forward.transpose(1, 2),
                weight=weight,
                bias=bias,
                activation=self.activation,
            ).transpose(1, 2)
        else:
            x_forward = self.act(self.conv1d(x_forward.transpose(1, 2)).transpose(1, 2))
            x_forward = x_forward[:, :seqlen, :]

        x_dbl_forward = self.x_proj(x_forward)
        dt_forward, B_forward, C_forward = torch.split(
            x_dbl_forward, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_forward = self.dt_proj(dt_forward)

        dt_forward = rearrange(dt_forward, "b l d -> b d l")
        B_forward = rearrange(B_forward, "b l n -> b n l").contiguous()
        C_forward = rearrange(C_forward, "b l n -> b n l").contiguous()
        x_forward = rearrange(x_forward, "b l d -> b d l")

        A_forward = -torch.exp(self.A_log.float()).to(x_forward.device)
        y_forward = selective_scan_fn(
            x_forward,
            dt_forward,
            A_forward,
            B_forward,
            C_forward,
            self.D.float().to(x_forward.device),
            z=None,
            delta_bias=self.dt_proj.bias.float().to(x_forward.device),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )

        if ssm_state is not None:
            y_forward, last_state = y_forward
            ssm_state.copy_(last_state)

        y_forward = rearrange(y_forward, "b d l -> b l d")
        y_forward = self.norm(y_forward, z)
        out = self.out_proj(y_forward)

        return out

    def _forward_bi_same(self, x: torch.Tensor, conv_state=None, ssm_state=None):
        batch, seqlen, dim = x.shape
        out_forward = self._forward_single(x, conv_state, ssm_state)
        x_rev = x.flip(1)

        xz_rev = self.in_proj(x_rev)
        x_backward, z_rev = xz_rev.chunk(2, dim=-1)

        if causal_conv1d_fn is not None and x_backward.is_cuda:
            weight = rearrange(self.conv1d_b.weight, "d 1 w -> d w").to(x_backward.device)
            bias = self.conv1d_b.bias.to(x_backward.device) if self.conv1d_b.bias is not None else None
            x_backward = causal_conv1d_fn(
                x=x_backward.transpose(1, 2),
                weight=weight,
                bias=bias,
                activation=self.activation,
            ).transpose(1, 2)
        else:
            x_backward = self.act(self.conv1d_b(x_backward.transpose(1, 2)).transpose(1, 2))
            x_backward = x_backward[:, :seqlen, :]

        x_dbl_backward = self.x_proj_b(x_backward)
        dt_backward, B_backward, C_backward = torch.split(
            x_dbl_backward, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_backward = self.dt_proj_b(dt_backward)

        dt_backward = rearrange(dt_backward, "b l d -> b d l")
        B_backward = rearrange(B_backward, "b l n -> b n l").contiguous()
        C_backward = rearrange(C_backward, "b l n -> b n l").contiguous()
        x_backward = rearrange(x_backward, "b l d -> b d l")

        A_backward = -torch.exp(self.A_b_log.float()).to(x_backward.device)
        y_backward = selective_scan_fn(
            x_backward,
            dt_backward,
            A_backward,
            B_backward,
            C_backward,
            self.D_b.float().to(x_backward.device),
            z=None,
            delta_bias=self.dt_proj_b.bias.float().to(x_backward.device),
            delta_softplus=True,
            return_last_state=False,
        )

        y_backward = rearrange(y_backward, "b d l -> b l d").flip(1)
        y_backward = self.norm(y_backward, z_rev.flip(1))
        out_backward = self.out_proj(y_backward)

        if self.if_devide_out:
            out = (out_forward + out_backward) / 2
        else:
            out = out_forward + out_backward

        return out

    def _forward_bi_cross(self, x_a: torch.Tensor, x_b: torch.Tensor, conv_state=None, ssm_state=None):
        batch, seqlen, dim = x_a.shape
        out_forward = self._forward_single(x_a, conv_state, ssm_state)
        x_b_rev = x_b.flip(1)

        xz_b_rev = self.in_proj(x_b_rev)
        x_backward, z_b_rev = xz_b_rev.chunk(2, dim=-1)

        if causal_conv1d_fn is not None and x_backward.is_cuda:
            weight = rearrange(self.conv1d_b.weight, "d 1 w -> d w").to(x_backward.device)
            bias = self.conv1d_b.bias.to(x_backward.device) if self.conv1d_b.bias is not None else None
            x_backward = causal_conv1d_fn(
                x=x_backward.transpose(1, 2),
                weight=weight,
                bias=bias,
                activation=self.activation,
            ).transpose(1, 2)
        else:
            x_backward = self.act(self.conv1d_b(x_backward.transpose(1, 2)).transpose(1, 2))
            x_backward = x_backward[:, :seqlen, :]

        x_dbl_backward = self.x_proj_b(x_backward)
        dt_backward, B_backward, C_backward = torch.split(
            x_dbl_backward, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt_backward = self.dt_proj_b(dt_backward)

        dt_backward = rearrange(dt_backward, "b l d -> b d l")
        B_backward = rearrange(B_backward, "b l n -> b n l").contiguous()
        C_backward = rearrange(C_backward, "b l n -> b n l").contiguous()
        x_backward = rearrange(x_backward, "b l d -> b d l")

        A_backward = -torch.exp(self.A_b_log.float()).to(x_backward.device)
        y_backward = selective_scan_fn(
            x_backward,
            dt_backward,
            A_backward,
            B_backward,
            C_backward,
            self.D_b.float().to(x_backward.device),
            z=None,
            delta_bias=self.dt_proj_b.bias.float().to(x_backward.device),
            delta_softplus=True,
            return_last_state=False,
        )

        y_backward = rearrange(y_backward, "b d l -> b l d").flip(1)
        y_backward = self.norm(y_backward, z_b_rev.flip(1))
        out_backward = self.out_proj(y_backward)

        if self.if_devide_out:
            out = (out_forward + out_backward) / 2
        else:
            out = out_forward + out_backward

        return out

    def _forward_bi_swap(self, x_a: torch.Tensor, x_b: torch.Tensor, conv_state=None, ssm_state=None):
        return self._forward_bi_cross(x_b, x_a, conv_state, ssm_state)

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "仅支持单token推理"

        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        if causal_conv1d_update is None or not x.is_cuda:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D,
                z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        y = self.norm(y.unsqueeze(1), z.unsqueeze(1)).squeeze(1)
        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv,
            device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state,
            device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        self.layer_idx = getattr(self, 'layer_idx', 0)
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class CrossMambaFusion(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            if_devide_out=False,
            in_channels=None,
            out_channels=None,
            device=None,
            dtype=None,
    ):
        super().__init__()

        self.mamba1 = CrossMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            cross_mode='bi_cross',
            if_devide_out=if_devide_out,
            in_channels=in_channels,
            out_channels=d_model,
            device=device,
            dtype=dtype,
        )

        self.mamba2 = CrossMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            cross_mode='bi_swap',
            if_devide_out=if_devide_out,
            in_channels=in_channels,
            out_channels=d_model,
            device=device,
            dtype=dtype,
        )

        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        self.use_feat_map = (in_channels is not None) and (out_channels is not None)
        if self.use_feat_map:
            self.final_decoder = nn.Conv2d(
                in_channels=d_model,
                out_channels=out_channels,
                kernel_size=1,
                bias=bias,
                device=device,
                dtype=dtype
            )

    def forward(self, visible: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        out1 = self.mamba1(visible, infrared)
        out2 = self.mamba2(visible, infrared)

        if self.use_feat_map and visible.dim() == 4:
            B, C, H, W = out1.shape
            out1_seq = rearrange(out1, 'b d h w -> b (h w) d')
            out2_seq = rearrange(out2, 'b d h w -> b (h w) d')
            fused_seq = torch.cat([out1_seq, out2_seq], dim=-1)
            fused_seq = self.fusion_proj(fused_seq)
            fused = rearrange(fused_seq, 'b (h w) d -> b d h w', h=H, w=W)
            fused = self.final_decoder(fused)
        else:
            fused = torch.cat([out1, out2], dim=-1)
            fused = self.fusion_proj(fused)

        return fused

TEST_CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "d_model": 64,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "in_channels": 16,
    "out_channels": 32,
    "batch_size": 8,
    "seq_len": 1024,
    "feat_h": 32,
    "feat_w": 32,
}

def test_cross_mamba_single(mode: str, input_type: str):
    cfg = TEST_CONFIG
    device = cfg["device"]
    d_model = cfg["d_model"]
    batch_size = cfg["batch_size"]
    in_channels = cfg["in_channels"]
    out_channels = cfg["out_channels"]
    seq_len = cfg["seq_len"]
    feat_h = cfg["feat_h"]
    feat_w = cfg["feat_w"]

    mamba = CrossMamba(
        d_model=d_model,
        d_state=cfg["d_state"],
        d_conv=cfg["d_conv"],
        expand=cfg["expand"],
        cross_mode=mode,
        in_channels=in_channels if input_type == "feat" else None,
        out_channels=out_channels if input_type == "feat" else None,
        device=device,
    ).to(device)

    if input_type == "seq":
        x_a = torch.randn(batch_size, seq_len, d_model, device=device)
        x_b = torch.randn(batch_size, seq_len, d_model, device=device) if mode in ["bi_cross", "bi_swap"] else None
    else:
        x_a = torch.randn(batch_size, in_channels, feat_h, feat_w, device=device)
        x_b = torch.randn(batch_size, in_channels, feat_h, feat_w, device=device) if mode in ["bi_cross", "bi_swap"] else None

    try:
        with torch.no_grad():
            output = mamba(x_a, x_b)
    except ValueError as e:
        if mode in ["bi_cross", "bi_swap"] and x_b is None:
            print(f"⚠ 预期错误：模式 {mode} 要求两个输入，触发 ValueError")
            return True
        else:
            raise e

    if input_type == "seq":
        assert output.shape == (batch_size, seq_len, d_model), \
            f"CrossMamba {mode} ({input_type}) 输出形状错误，预期 {(batch_size, seq_len, d_model)}，实际 {output.shape}"
    else:
        assert output.shape == (batch_size, out_channels, feat_h, feat_w), \
            f"CrossMamba {mode} ({input_type}) 输出形状错误，预期 {(batch_size, out_channels, feat_h, feat_w)}，实际 {output.shape}"

    print(f"✓ CrossMamba {mode} ({input_type}) 测试通过")
    return True

def test_cross_mamba_all():
    modes = ["single", "bi_same", "bi_cross", "bi_swap"]
    input_types = ["seq", "feat"]

    for mode in modes:
        for input_type in input_types:
            test_cross_mamba_single(mode, input_type)

    print("="*50)
    print("✓ CrossMamba 所有模式测试通过")
    print("="*50)

def test_cross_mamba_fusion_all():
    cfg = TEST_CONFIG
    device = cfg["device"]
    d_model = cfg["d_model"]
    batch_size = cfg["batch_size"]
    in_channels = cfg["in_channels"]
    out_channels = cfg["out_channels"]
    seq_len = cfg["seq_len"]
    feat_h = cfg["feat_h"]
    feat_w = cfg["feat_w"]

    fusion_seq = CrossMambaFusion(
        d_model=d_model,
        d_state=cfg["d_state"],
        d_conv=cfg["d_conv"],
        expand=cfg["expand"],
        in_channels=None,
        out_channels=None,
        device=device,
    ).to(device)

    visible_seq = torch.randn(batch_size, seq_len, d_model, device=device)
    infrared_seq = torch.randn(batch_size, seq_len, d_model, device=device)

    with torch.no_grad():
        fused_seq = fusion_seq(visible_seq, infrared_seq)

    assert fused_seq.shape == (batch_size, seq_len, d_model), \
        f"CrossMambaFusion 序列输出形状错误，预期 {(batch_size, seq_len, d_model)}，实际 {fused_seq.shape}"
    print("✓ CrossMambaFusion 序列输入测试通过")

    fusion_feat = CrossMambaFusion(
        d_model=d_model,
        d_state=cfg["d_state"],
        d_conv=cfg["d_conv"],
        expand=cfg["expand"],
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
    ).to(device)

    visible_feat = torch.randn(batch_size, in_channels, feat_h, feat_w, device=device)
    infrared_feat = torch.randn(batch_size, in_channels, feat_h, feat_w, device=device)

    with torch.no_grad():
        fused_feat = fusion_feat(visible_feat, infrared_feat)

    assert fused_feat.shape == (batch_size, out_channels, feat_h, feat_w), \
        f"CrossMambaFusion 特征图输出形状错误，预期 {(batch_size, out_channels, feat_h, feat_w)}，实际 {fused_feat.shape}"
    print("✓ CrossMambaFusion 特征图输入测试通过")

    print("="*50)
    print("✓ CrossMambaFusion 所有测试通过")
    print("="*50)

def test_gradient_flow():
    cfg = TEST_CONFIG
    device = cfg["device"]
    d_model = cfg["d_model"]
    batch_size = cfg["batch_size"]

    mamba = CrossMamba(
        d_model=d_model,
        cross_mode="bi_cross",
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        device=device,
    ).to(device)

    x_a = torch.randn(batch_size, cfg["in_channels"], cfg["feat_h"], cfg["feat_w"], device=device, requires_grad=True)
    x_b = torch.randn(batch_size, cfg["in_channels"], cfg["feat_h"], cfg["feat_w"], device=device, requires_grad=True)

    output = mamba(x_a, x_b)
    loss = output.sum()
    loss.backward()

    assert x_a.grad is not None and x_a.grad.sum() != 0, "输入A的梯度未传播"
    assert x_b.grad is not None and x_b.grad.sum() != 0, "输入B的梯度未传播"
    assert any(p.grad is not None and p.grad.sum() != 0 for p in mamba.parameters()), "模型参数梯度未传播"

    print("✓ 梯度传播测试通过")
    print("="*50)

def run_all_tests():
    print("开始运行 CrossMamba 测试套件...")
    print("="*50)

    test_cross_mamba_all()

    test_cross_mamba_fusion_all()

    test_gradient_flow()

    print("🎉 所有测试全部通过！")

if __name__ == "__main__":
    run_all_tests()