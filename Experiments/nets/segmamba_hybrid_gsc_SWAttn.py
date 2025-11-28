# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import torch.nn as nn
import torch 
from functools import partial
import math 

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 
from timm.models.layers import trunc_normal_

from .kan_fJNB import KAN
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x
    
class FKANMLP(nn.Module):
    """
    Simple KAN-based MLP for token features.
    Input / output: (B, N, C)
    """
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.kan = KAN(
            layers_hidden=[dim, mlp_dim, dim],
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        x = self.norm(x)
        x_flat = x.reshape(B * N, C)
        y_flat = self.kan(x_flat)
        y = y_flat.view(B, N, C)
        y = self.dropout(y)
        return y



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


#     def forward(self, x):
#         b,c,h,w = x.shape

#         qkv = self.qkv_dwconv(self.qkv(x))
#         q,k,v = qkv.chunk(3, dim=1)   
        
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)
        
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out


# class TokenMDTA(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.inner = Attention(dim, num_heads, bias)

#     def forward(self, x):
#         # x: (B, N, D)
#         B, N, D = x.shape
#         H = W = int(math.sqrt(N))
#         assert H * W == N, "Token count N must be a perfect square"

#         x_2d = x.permute(0, 2, 1).reshape(B, D, H, W)   # (B, D, H, W)
#         out_2d = self.inner(x_2d)                       # (B, D, H, W)
#         out = out_2d.reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)

#         # no explicit attention weights here
#         weights = None
#         return out, weights


class MambaVisionMixer(nn.Module):
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
        use_fast_path=True, 
        layer_idx=None,
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
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def _check_tensor(self, name, x):
        if x is None:
            raise RuntimeError(f"[MambaVisionMixer] {name} is None")
        if not torch.isfinite(x).all():
            raise RuntimeError(
                f"[MambaVisionMixer] Non-finite values in {name}: "
                f"min={x.min().item()}, max={x.max().item()}"
            )
    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        if hidden_states.dim() != 3:
            raise RuntimeError(
                f"[MambaVisionMixer] Expected (B, L, D), got {hidden_states.shape}"
            )
        
        B, seqlen, D = hidden_states.shape
        if D != self.d_model:
            raise RuntimeError(
                f"[MambaVisionMixer] d_model mismatch: got {D}, expected {self.d_model}"
            )

        self._check_tensor("hidden_states", hidden_states)

        # xz = self.in_proj(hidden_states)
        # xz = rearrange(xz, "b l d -> b d l")
        # x, z = xz.chunk(2, dim=1)
        # A = -torch.exp(self.A_log.float())
        # x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        # z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        # x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        # dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        # B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        # y = selective_scan_fn(x, 
        #                       dt, 
        #                       A, 
        #                       B, 
        #                       C, 
        #                       self.D.float(), 
        #                       z=None, 
        #                       delta_bias=self.dt_proj.bias.float(), 
        #                       delta_softplus=True, 
        #                       return_last_state=None)
        
        # y = torch.cat([y, z], dim=1)
        # y = rearrange(y, "b d l -> b l d")
        # out = self.out_proj(y)
        # return out
        try:
            xz = self.in_proj(hidden_states)          # (B, L, d_inner)
            self._check_tensor("xz", xz)

            xz = rearrange(xz, "b l d -> b d l")      # (B, d_inner, L)
            x, z = xz.chunk(2, dim=1)                # each (B, d_inner/2, L)
            self._check_tensor("x_before_conv", x)
            self._check_tensor("z_before_conv", z)

            A = -torch.exp(self.A_log.float())        # (d_inner/2, d_state)

            x = F.silu(
                F.conv1d(
                    input=x,
                    weight=self.conv1d_x.weight,
                    bias=self.conv1d_x.bias,
                    padding="same",
                    groups=self.d_inner // 2,
                )
            )
            z = F.silu(
                F.conv1d(
                    input=z,
                    weight=self.conv1d_z.weight,
                    bias=self.conv1d_z.bias,
                    padding="same",
                    groups=self.d_inner // 2,
                )
            )

            self._check_tensor("x_after_conv", x)
            self._check_tensor("z_after_conv", z)

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            self._check_tensor("x_dbl", x_dbl)

            dt, Bmat, Cmat = torch.split(
                x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
            )

            dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
            Bmat = rearrange(Bmat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            Cmat = rearrange(Cmat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            self._check_tensor("dt", dt)
            self._check_tensor("Bmat", Bmat)
            self._check_tensor("Cmat", Cmat)

            y = selective_scan_fn(
                x,
                dt,
                A,
                Bmat,
                Cmat,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )

            self._check_tensor("y_after_scan", y)

            y = torch.cat([y, z], dim=1)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            self._check_tensor("out", out)
            return out

        except RuntimeError as e:
            # Catch low-level CUDA assert and re-raise with context
            if "device-side assert" in str(e).lower():
                raise RuntimeError(
                    "[MambaVisionMixer] CUDA device-side assert inside selective_scan_fn "
                    f"or conv1d.\n"
                    f"hidden_states shape: {hidden_states.shape}, "
                    f"d_model={self.d_model}, d_inner={self.d_inner}, "
                    f"d_state={self.d_state}, seq_len={seqlen}"
                ) from e
            else:
                raise


class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class WindowTokenAttention(nn.Module):
    """
    Swin-style window attention on token features.
    Input / output: (B, N, C)
    Assumes N = H * W and H, W are multiples of window_size.
    """
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        # Reuse Swin's WindowAttention
        self.inner = WindowAttention(
            dim=dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        # infer H, W assuming square feature map and depth=1
        H = W = int(N ** 0.5)
        assert H * W == N, f"WindowTokenAttention: N={N} is not a perfect square"

        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"H={H}, W={W} not divisible by window_size={self.window_size}"

        # (B, N, C) -> (B, H, W, C)
        x_2d = x.view(B, H, W, C)

        # partition windows: (nW*B, ws, ws, C)
        x_windows = window_partition(x_2d, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # attention per window
        attn_windows = self.inner(x_windows, mask=None)  # (nW*B, ws*ws, C)

        # merge windows back: (nW*B, ws, ws, C) → (B, H, W, C)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x_2d = window_reverse(attn_windows, self.window_size, H, W)

        # (B, H*W, C)
        out = x_2d.view(B, N, C)
        return out, None

class TransformerMambaBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0,
                 d_state=8, d_conv=3, expand=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        mlp_dim = int(dim * mlp_ratio)

        # --- Transformer part ---
        print("At Transformer ")
        self.ln1 = nn.LayerNorm(dim)      # for MDTA
        # self.attn = TokenMDTA(dim=dim, num_heads=num_heads, bias=True)
        print("At Shifted window attention ")
        self.attn = WindowTokenAttention(
                        dim=dim,
                        num_heads=num_heads,
                        window_size=7,      # or 4, 8, etc. See note below.
                        qkv_bias=True,
                    )
        self.ln2 = nn.LayerNorm(dim)      # for first f-KAN
        self.ffn1 = FKANMLP(dim, mlp_dim)

        # --- Mamba part ---
        print("At MambaVisionMixer ")
        self.ln3 = nn.LayerNorm(dim)      # for VSSM
        self.vssm = MambaVisionMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.ln4 = nn.LayerNorm(dim)      # for second f-KAN
        self.ffn2 = FKANMLP(dim, mlp_dim)

    # OLD DEPRECATED
    # def forward(self, x5d):
    #     # x5d: (B, C, D, H, W)
    #     B, C = x5d.shape[:2]
    #     D, H, W = x5d.shape[2:]
    #     N = D * H * W

    #     # ===== flatten to tokens =====
    #     x = x5d.view(B, C, N).transpose(1, 2)   # (B, N, C)
    #     x_in = x                                # original input tokens

    #     # ================= TRANSFORMER PART =================
    #     # 1) LN -> MDTA -> add residual (orig input)
    #     t = self.ln1(x_in)
    #     t, _ = self.attn(t)                     # (B, N, C)
    #     t = x_in + t                            # attn_residual

    #     # 2) LN -> f-KAN -> add residual (orig input)
    #     u = self.ln2(t)
    #     u = self.ffn1(u)                        # (B, N, C)
    #     x_tr = x_in + u                         # transformer output

    #     # ================== MAMBA PART =====================
    #     # 3) LN -> VSSM -> add residual (transformer output)
    #     m = self.ln3(x_tr)
    #     m = self.vssm(m)                        # (B, N, C)
    #     m = x_tr + m                            # mamba_residual

    #     # 4) LN -> f-KAN -> add residual (transformer output)
    #     n = self.ln4(m)
    #     n = self.ffn2(n)                        # (B, N, C)
    #     x_out_tokens = x_tr + n                 # final output tokens

    #     # ===== back to 5D =====
    #     x_out = x_out_tokens.transpose(1, 2).view(B, C, D, H, W)
    #     return x_out

    def forward(self, x5d):
        # x5d: (B, C, D, H, W)
        B, C = x5d.shape[:2]
        D, H, W = x5d.shape[2:]
        N = D * H * W

        # ===== flatten to tokens =====
        x = x5d.view(B, C, N).transpose(1, 2)   # (B, N, C)
        x_in = x                                # original input tokens

        # ================= TRANSFORMER PART =================
        # 1) LN -> MDTA -> add residual (orig input)
        t = self.ln1(x_in)
        t, _ = self.attn(t)                     # (B, N, C)
        t = x_in + t                            # attn_residual

        # 2) LN -> f-KAN -> add residual (orig input)
        u = self.ln2(t)
        u = self.ffn1(u)                        # (B, N, C)
        u = u + t                            # f-KAN residual
        x_tr = x_in + u                         # transformer output

        # ================== MAMBA PART =====================
        # 3) LN -> VSSM -> add residual (transformer output)
        m = self.ln3(x_tr)
        m = self.vssm(m)                        # (B, N, C)
        m = x_tr + m                            # mamba_residual

        # 4) LN -> f-KAN -> add residual (transformer output)
        n = self.ln4(m)
        n = self.ffn2(n)                        # (B, N, C)
        n = n + m                            # f-KAN residual
        x_out_tokens = x_tr + n                 # final output tokens

        # ===== back to 5D =====
        x_out = x_out_tokens.transpose(1, 2).view(B, C, D, H, W)
        return x_out

class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        self.nonliner2 = nn.ReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        self.nonliner3 = nn.ReLU()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        self.nonliner4 = nn.ReLU()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual
    
class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_chans=1,
        depths=[2, 2, 2, 2],
        dims=[48, 96, 192, 384],
        num_heads=4,
        mlp_ratio=4.0,
        d_state=8,
        d_conv=3,
        expand=1,
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.0,         
        layer_scale_init_value=1e-6,
        
    ):
        super().__init__()

        self.out_indices = out_indices

        # ---------- Downsampling path (kept from old encoder) ----------
        self.downsample_layers = nn.ModuleList()

        # Stem: Conv3d with (1,7,7) and stride (1,2,2)
        stem = nn.Sequential(
            nn.Conv3d(
                in_chans,
                dims[0],
                kernel_size=(1, 7, 7),
                stride=(1, 2, 2),
                padding=(0, 3, 3),
            ),
        )
        self.downsample_layers.append(stem)

        # Next 3 downsample layers: only H,W downsampled
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=(1, 2, 2),
                    stride=(1, 2, 2),
                ),
            )
            self.downsample_layers.append(downsample_layer)

        # ---------- Per-stage Transformer+KAN+VSSM blocks ----------
        self.stages = nn.ModuleList()
        print("at gsc")
        self.gscs = nn.ModuleList()
        for i in range(4):
            gsc = GSC(dims[i])
            stage = nn.Sequential(
                *[
                    TransformerMambaBlock(
                        dim=dims[i],
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                    for _ in range(depths[i])
                ]
            )
            self.gscs.append(gsc)
            # cur += depths[i]
            self.stages.append(stage)

    def forward_features(self, x):
        """
        x: (B, in_chans, D, H, W)

        After stage 0: (B, dims[0], D, H/2,  W/2)
        After stage 1: (B, dims[1], D, H/4,  W/4)
        After stage 2: (B, dims[2], D, H/8,  W/8)
        After stage 3: (B, dims[3], D, H/16, W/16)
        """
        outs = []
        for i in range(4):
            # 3D downsampling (unchanged)
            x = self.downsample_layers[i](x)

            x = self.gscs[i](x)
            # Token pipeline block(s) on flattened tokens
            x = self.stages[i](x)

            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=1,   # <- you can change default 13 to 1 if most of your work is binary
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        # print("Initializing SegMamba")
        print("Initializing SegMamba with Hybrid Encoder along with GSC with Shifted Window attention")
        self.spatial_dims = spatial_dims
        self.vit = MambaEncoder(in_chans, 
                                depths=depths,
                                dims=feat_size,
                                drop_path_rate=drop_path_rate,
                                layer_scale_init_value=layer_scale_init_value,
                              )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            # upsample_kernel_size=2,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            # upsample_kernel_size=2,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            # upsample_kernel_size=2,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            # upsample_kernel_size=2,
            upsample_kernel_size=(1, 2, 2),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    # def forward(self, x_in):
    #     outs = self.vit(x_in)
    #     enc1 = self.encoder1(x_in)
    #     x2 = outs[0]
    #     enc2 = self.encoder2(x2)
    #     x3 = outs[1]
    #     enc3 = self.encoder3(x3)
    #     x4 = outs[2]
    #     enc4 = self.encoder4(x4)
    #     enc_hidden = self.encoder5(outs[3])
    #     dec3 = self.decoder5(enc_hidden, enc4)
    #     dec2 = self.decoder4(dec3, enc3)
    #     dec1 = self.decoder3(dec2, enc2)
    #     dec0 = self.decoder2(dec1, enc1)
    #     out = self.decoder1(dec0)
                
    #     return self.out(out)

    def _check_numerics(self, name, x):
        if not torch.isfinite(x).all():
            # This will raise before any CUDA kernel does something nasty
            raise RuntimeError(
                f"[SegMamba] Non-finite values in {name}: "
                f"min={x.min().item()}, max={x.max().item()}"
            )
    
    def forward(self, x_in):
        """
        x_in: [B, C, H, W] or [B, C, D, H, W]
        """
        squeeze_depth = False

        # ---- Input checks ----
        if x_in.dim() not in (4, 5):
            raise RuntimeError(
                f"[SegMamba] Expected 4D or 5D input, got {x_in.dim()}D with shape {x_in.shape}"
            )

        if x_in.dim() == 4:
            x_in = x_in.unsqueeze(2)   # [B, C, 1, H, W]
            squeeze_depth = True

        if x_in.size(1) != self.in_chans:
            raise RuntimeError(
                f"[SegMamba] Channel mismatch: got {x_in.size(1)} channels, "
                f"expected {self.in_chans}"
            )

        self._check_numerics("x_in", x_in)

        # --- Encoder path with Mamba features ---
        outs = self.vit(x_in)        # tuple of 4 feature maps
        for i, o in enumerate(outs):
            if o is None:
                raise RuntimeError(f"[SegMamba] vit outs[{i}] is None")
            self._check_numerics(f"vit outs[{i}]", o)

        enc1 = self.encoder1(x_in)
        self._check_numerics("enc1", enc1)

        x2 = outs[0]
        enc2 = self.encoder2(x2)
        self._check_numerics("enc2", enc2)

        x3 = outs[1]
        enc3 = self.encoder3(x3)
        self._check_numerics("enc3", enc3)

        x4 = outs[2]
        enc4 = self.encoder4(x4)
        self._check_numerics("enc4", enc4)

        enc_hidden = self.encoder5(outs[3])
        self._check_numerics("enc_hidden", enc_hidden)

        # --- Decoder path ---
        dec3 = self.decoder5(enc_hidden, enc4)
        self._check_numerics("dec3", dec3)

        dec2 = self.decoder4(dec3, enc3)
        self._check_numerics("dec2", dec2)

        dec1 = self.decoder3(dec2, enc2)
        self._check_numerics("dec1", dec1)

        dec0 = self.decoder2(dec1, enc1)
        self._check_numerics("dec0", dec0)

        out = self.decoder1(dec0)
        self._check_numerics("decoder1_out", out)

        out = self.out(out)
        self._check_numerics("out_logits_5d", out)

        if squeeze_depth:
            out = out.squeeze(2)  # [B, out_chans, H, W]
            self._check_numerics("out_logits_4d", out)

        return out
    
    # def forward(self, x_in):
    #     """
    #     x_in comes from your ACC-UNet pipeline as [B, C, H, W].

    #     We:
    #     - unsqueeze a fake depth dim -> [B, C, 1, H, W]
    #     - run the 3D Mamba encoder + UNETR blocks
    #     - squeeze depth back -> [B, out_chans, H, W]
    #     """
    #     squeeze_depth = False
    #     print("[SegMamba] x_in:", x_in.shape) 

    #     if x_in.dim() == 4:
    #         # [B, C, H, W] -> [B, C, 1, H, W]
    #         x_in = x_in.unsqueeze(2)
    #         squeeze_depth = True
    #     print("[SegMamba] x_in after unsqueeze:", x_in.shape)

    #     # --- Encoder path with Mamba features as in your original code ---
    #     outs = self.vit(x_in)        # tuple of 4 feature maps
    #     for i, f in enumerate(outs):
    #         print(f"[SegMamba] vit outs[{i}]:", f.shape)

    #     enc1 = self.encoder1(x_in)   # skip at full res
    #     print("[SegMamba] enc1:", enc1.shape)

    #     x2 = outs[0]
    #     enc2 = self.encoder2(x2)
    #     print("[SegMamba] enc2:", enc2.shape)

    #     x3 = outs[1]
    #     enc3 = self.encoder3(x3)
    #     print("[SegMamba] enc3:", enc3.shape)

    #     x4 = outs[2]
    #     enc4 = self.encoder4(x4)
    #     print("[SegMamba] enc4:", enc4.shape)

    #     enc_hidden = self.encoder5(outs[3])
    #     print("[SegMamba] enc_hidden:", enc_hidden.shape)

    #     # --- Decoder path ---
    #     dec3 = self.decoder5(enc_hidden, enc4)
    #     print("[SegMamba] dec3:", dec3.shape)

    #     dec2 = self.decoder4(dec3, enc3)
    #     print("[SegMamba] dec2:", dec2.shape)

    #     dec1 = self.decoder3(dec2, enc2)
    #     print("[SegMamba] dec1:", dec1.shape)

    #     dec0 = self.decoder2(dec1, enc1)
    #     print("[SegMamba] dec0:", dec0.shape)

    #     out = self.decoder1(dec0)     # [B, C, D, H, W]
    #     print("[SegMamba] out before final conv:", out.shape)

    #     out = self.out(out)           # [B, out_chans, D, H, W]
    #     print("[SegMamba] out after final conv:", out.shape)

    #     if squeeze_depth:
    #         out = out.squeeze(2)      # [B, out_chans, H, W]
    #         print("[SegMamba] out after squeeze depth:", out.shape)

    #     return out
    

    # def forward(self, x_in):
    #     """
    #     x_in comes from your ACC-UNet pipeline as [B, C, H, W].

    #     We:
    #     - unsqueeze a fake depth dim -> [B, C, 1, H, W]
    #     - run the 3D Mamba encoder + UNETR blocks
    #     - squeeze depth back -> [B, out_chans, H, W]
    #     """
    #     squeeze_depth = False
    #     if x_in.dim() == 4:
    #         # [B, C, H, W] -> [B, C, 1, H, W]
    #         x_in = x_in.unsqueeze(2)
    #         squeeze_depth = True

    #     # --- Encoder path with Mamba features as in your original code ---
    #     outs = self.vit(x_in)        # tuple of 4 feature maps
    #     enc1 = self.encoder1(x_in)   # skip at full res

    #     x2 = outs[0]
    #     enc2 = self.encoder2(x2)

    #     x3 = outs[1]
    #     enc3 = self.encoder3(x3)

    #     x4 = outs[2]
    #     enc4 = self.encoder4(x4)

    #     enc_hidden = self.encoder5(outs[3])

    #     # --- Decoder path ---
    #     dec3 = self.decoder5(enc_hidden, enc4)
    #     dec2 = self.decoder4(dec3, enc3)
    #     dec1 = self.decoder3(dec2, enc2)
    #     dec0 = self.decoder2(dec1, enc1)
    #     out = self.decoder1(dec0)     # [B, C, D, H, W]

    #     out = self.out(out)           # [B, out_chans, D, H, W]

    #     if squeeze_depth:
    #         out = out.squeeze(2)      # [B, out_chans, H, W]

    #     return out

    
