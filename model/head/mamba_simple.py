import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined


class mamba2(nn.Module):
    def __init__(
            self,
            d_model,  # model dimension / input feature dimension
            d_state=16,
            d_conv=4,
            conv_init=None,
            expand=2,  # expansion factor
            headdim=32,  # dimension of each head
            ngroups=1,  # number of groups
            A_init_range=(1, 16),  # initialization range for matrix A
            dt_min=0.001,  # minimum value of time step parameter
            dt_max=0.1,  # maximum value of time step parameter
            dt_init_floor=1e-4,  # initialization floor for time step parameter
            dt_limit=(0.0, float("inf")),  # limit range for time step parameter
            learnable_init_states=False,  # whether to use learnable initial states
            activation="swish",  # type of activation function
            bias=False,  # whether to use bias in linear layers
            conv_bias=True,  # whether to use bias in convolution layers
            # fused kernel and chunk options
            chunk_size=256,  # chunk size
            use_mem_eff_path=False,  # whether to use memory-efficient path
            layer_idx=None,  # layer index (for universal modules)
            device=None,  # device
            dtype=None,  # data type
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # save parameters to instance variables
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = self.expand * self.d_model  # inner dimension = expansion factor × model dimension
        self.headdim = headdim
        self.ngroups = ngroups
        # ensure inner dimension is divisible by head dimension
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim  # number of heads = inner dimension ÷ head dimension
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        # input projection layer: total dimension of five parts [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        # convolution dimension = inner dimension + 2 × number of groups × state dimension
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        # 1D convolution layer, using grouped convolution for efficient computation
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,  # depth-wise separable convolution
            padding=d_conv - 1,  # padding for causal convolution
            **factory_kwargs,
        )
        # if convolution initialization range is specified, perform uniform initialization
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # mark convolution weights to be excluded from weight decay (commented)
        # self.conv1d.weight._no_weight_decay = True

        # if using learnable initial states, create parameters
        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True  # exclude from weight decay

        # SiLU activation function
        self.act = nn.SiLU()

        # initialize bias for time step parameter (dt)
        # randomly initialize dt within specified range
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)  # ensure not less than the floor
        # compute inverse of softplus for bias initialization
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # mark to be excluded from weight decay
        self.dt_bias._no_weight_decay = True

        # initialization for parameter A (state transition matrix)
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        # uniformly initialize A within specified range
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)  # use log form to ensure positivity
        self.A_log = nn.Parameter(A_log)
        # mark to be excluded from weight decay
        self.A_log._no_weight_decay = True

        # parameter D (skip connection weights), initialized to all ones
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True  # mark to be excluded from weight decay

        # additional normalization layer before output projection
        assert RMSNormGated is not None  # ensure RMSNormGated is imported
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)

        # output projection layer
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        Forward propagation function
        u: (B, L, D) input tensor, batch size × sequence length × feature dimension
        Returns: output with the same shape as u and gating signal z
        """
        # --------------------------- 关键修改1：确保输入张量内存连续 ---------------------------
        u = u.contiguous()
        batch, seqlen, dim = u.shape

        # input projection, get concatenation of z, x, B, C and dt
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        # --------------------------- 关键修改2：确保投影后的张量连续 ---------------------------
        zxbcdt = zxbcdt.contiguous()

        # extract z in advance to ensure definition in all paths
        z = zxbcdt[..., :self.d_inner]  # extract gating signal part
        # --------------------------- 关键修改3：确保z张量连续 ---------------------------
        z = z.contiguous()

        # compute matrix A (state transition matrix), ensure negativity
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        # prepare initial states
        initial_states = repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        # prepare dt limit parameters
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # select computation path based on flag
        if self.use_mem_eff_path:
            # use memory-efficient path (fully fused operation)
            out = mamba_split_conv1d_scan_combined(
                zxbcdt,  # input projection result
                # --------------------------- 关键修改4：确保卷积权重张量连续 ---------------------------
                rearrange(self.conv1d.weight, "d 1 w -> d w").contiguous(),
                self.conv1d.bias,  # convolution bias
                self.dt_bias,  # time step bias
                A,  # state transition matrix
                D=self.D,  # skip connection weights
                chunk_size=self.chunk_size,  # chunk size
                seq_idx=seq_idx,  # sequence index (optional)
                activation=self.activation,  # activation function
                rmsnorm_weight=self.norm.weight,  # normalization layer weights
                rmsnorm_eps=self.norm.eps,  # normalization layer epsilon
                outproj_weight=self.out_proj.weight,  # output projection weights
                outproj_bias=self.out_proj.bias,  # output projection bias
                headdim=self.headdim,  # head dimension
                ngroups=self.ngroups,  # number of groups
                norm_before_gate=False,  # normalization order
                initial_states=initial_states,  # initial states
                **dt_limit_kwargs,  # dt limit parameters
            )
        else:
            # non-fused path (for debugging or alternative implementation)
            # split input projection result into z, xBC and dt parts
            # note: since we have extracted z in advance, only need to extract xBC and dt
            xBC_dt = zxbcdt[..., self.d_inner:]  # skip z part
            # --------------------------- 关键修改5：确保xBC_dt张量连续 ---------------------------
            xBC_dt = xBC_dt.contiguous()

            xBC, dt = torch.split(
                xBC_dt, [self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
            # --------------------------- 关键修改6：确保xBC和dt张量连续 ---------------------------
            xBC = xBC.contiguous()
            dt = dt.contiguous()

            # compute time step parameter
            dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
            assert self.activation in ["silu", "swish"]  # ensure activation function is supported

            # 1D convolution operation
            if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
                # use standard convolution + activation
                # --------------------------- 关键修改7：Transpose后强制连续 ---------------------------
                xBC_trans = xBC.transpose(1, 2).contiguous()
                xBC = self.act(self.conv1d(xBC_trans).transpose(1, 2).contiguous())
                xBC = xBC[:, :seqlen, :]  # crop to original sequence length (causal convolution)
            else:
                # use optimized causal convolution function
                # --------------------------- 关键修改8：修复causal_conv1d的输入布局 ---------------------------
                xBC_trans = xBC.transpose(1, 2).contiguous()  # 强制连续，解决stride问题
                xBC = causal_conv1d_fn(
                    x=xBC_trans,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w").contiguous(),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2).contiguous()  # Transpose后再次强制连续

            # split convolution result into three branches X, B, C
            # these correspond to V, K, Q in SSM/attention duality
            x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
            # --------------------------- 关键修改9：确保x/B/C张量连续 ---------------------------
            x = x.contiguous()
            B = B.contiguous()
            C = C.contiguous()

            # perform state space model computation using chunked scan
            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous(),  # 强制连续
                dt,  # time step parameter
                A,  # state transition matrix
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups).contiguous(),  # 强制连续
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups).contiguous(),  # 强制连续
                chunk_size=self.chunk_size,  # chunk size
                D=self.D,  # skip connection weights
                z=None,  # z is None here, to be handled later
                seq_idx=seq_idx,  # sequence index
                initial_states=initial_states,  # initial states**dt_limit_kwargs,  # dt limit parameters
            )
            # recombine multi-head output into original shape
            y = rearrange(y, "b l h p -> b l (h p)").contiguous()  # 强制连续

            # apply gating and normalization
            y = self.norm(y, z)  # use z as gating signal
            # output projection
            out = self.out_proj(y).contiguous()  # 强制连续

        # ensure to return z (gating signal)
        return out, z