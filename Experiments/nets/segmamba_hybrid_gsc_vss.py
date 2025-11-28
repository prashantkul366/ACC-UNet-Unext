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
from typing import Optional, Callable

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 

from timm.models.layers import DropPath
from .kan_fJNB import KAN
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

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
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TokenMDTA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inner = Attention(dim, num_heads, bias)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, "Token count N must be a perfect square"

        x_2d = x.permute(0, 2, 1).reshape(B, D, H, W)   # (B, D, H, W)
        out_2d = self.inner(x_2d)                       # (B, D, H, W)
        out = out_2d.reshape(B, D, N).permute(0, 2, 1)  # (B, N, D)

        # no explicit attention weights here
        weights = None
        return out, weights


# class MambaVisionMixer(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         d_state=16,
#         d_conv=4,
#         expand=2,
#         dt_rank="auto",
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init="random",
#         dt_scale=1.0,
#         dt_init_floor=1e-4,
#         conv_bias=True,
#         bias=False,
#         use_fast_path=True, 
#         layer_idx=None,
#         device=None,
#         dtype=None,
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#         self.use_fast_path = use_fast_path
#         self.layer_idx = layer_idx
#         self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
#         self.x_proj = nn.Linear(
#             self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
#         )
#         self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
#         dt_init_std = self.dt_rank**-0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(self.dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#         dt = torch.exp(
#             torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             self.dt_proj.bias.copy_(inv_dt)
#         self.dt_proj.bias._no_reinit = True
#         A = repeat(
#             torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=self.d_inner//2,
#         ).contiguous()
#         A_log = torch.log(A)
#         self.A_log = nn.Parameter(A_log)
#         self.A_log._no_weight_decay = True
#         self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
#         self.D._no_weight_decay = True
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
#         self.conv1d_x = nn.Conv1d(
#             in_channels=self.d_inner//2,
#             out_channels=self.d_inner//2,
#             bias=conv_bias//2,
#             kernel_size=d_conv,
#             groups=self.d_inner//2,
#             **factory_kwargs,
#         )
#         self.conv1d_z = nn.Conv1d(
#             in_channels=self.d_inner//2,
#             out_channels=self.d_inner//2,
#             bias=conv_bias//2,
#             kernel_size=d_conv,
#             groups=self.d_inner//2,
#             **factory_kwargs,
#         )

#     def _check_tensor(self, name, x):
#         if x is None:
#             raise RuntimeError(f"[MambaVisionMixer] {name} is None")
#         if not torch.isfinite(x).all():
#             raise RuntimeError(
#                 f"[MambaVisionMixer] Non-finite values in {name}: "
#                 f"min={x.min().item()}, max={x.max().item()}"
#             )
#     def forward(self, hidden_states):
#         """
#         hidden_states: (B, L, D)
#         Returns: same shape as hidden_states
#         """
#         if hidden_states.dim() != 3:
#             raise RuntimeError(
#                 f"[MambaVisionMixer] Expected (B, L, D), got {hidden_states.shape}"
#             )
        
#         B, seqlen, D = hidden_states.shape
#         if D != self.d_model:
#             raise RuntimeError(
#                 f"[MambaVisionMixer] d_model mismatch: got {D}, expected {self.d_model}"
#             )

#         self._check_tensor("hidden_states", hidden_states)

#         # xz = self.in_proj(hidden_states)
#         # xz = rearrange(xz, "b l d -> b d l")
#         # x, z = xz.chunk(2, dim=1)
#         # A = -torch.exp(self.A_log.float())
#         # x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
#         # z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
#         # x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
#         # dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#         # dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
#         # B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
#         # C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
#         # y = selective_scan_fn(x, 
#         #                       dt, 
#         #                       A, 
#         #                       B, 
#         #                       C, 
#         #                       self.D.float(), 
#         #                       z=None, 
#         #                       delta_bias=self.dt_proj.bias.float(), 
#         #                       delta_softplus=True, 
#         #                       return_last_state=None)
        
#         # y = torch.cat([y, z], dim=1)
#         # y = rearrange(y, "b d l -> b l d")
#         # out = self.out_proj(y)
#         # return out
#         try:
#             xz = self.in_proj(hidden_states)          # (B, L, d_inner)
#             self._check_tensor("xz", xz)

#             xz = rearrange(xz, "b l d -> b d l")      # (B, d_inner, L)
#             x, z = xz.chunk(2, dim=1)                # each (B, d_inner/2, L)
#             self._check_tensor("x_before_conv", x)
#             self._check_tensor("z_before_conv", z)

#             A = -torch.exp(self.A_log.float())        # (d_inner/2, d_state)

#             x = F.silu(
#                 F.conv1d(
#                     input=x,
#                     weight=self.conv1d_x.weight,
#                     bias=self.conv1d_x.bias,
#                     padding="same",
#                     groups=self.d_inner // 2,
#                 )
#             )
#             z = F.silu(
#                 F.conv1d(
#                     input=z,
#                     weight=self.conv1d_z.weight,
#                     bias=self.conv1d_z.bias,
#                     padding="same",
#                     groups=self.d_inner // 2,
#                 )
#             )

#             self._check_tensor("x_after_conv", x)
#             self._check_tensor("z_after_conv", z)

#             x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
#             self._check_tensor("x_dbl", x_dbl)

#             dt, Bmat, Cmat = torch.split(
#                 x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
#             )

#             dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
#             Bmat = rearrange(Bmat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
#             Cmat = rearrange(Cmat, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

#             self._check_tensor("dt", dt)
#             self._check_tensor("Bmat", Bmat)
#             self._check_tensor("Cmat", Cmat)

#             y = selective_scan_fn(
#                 x,
#                 dt,
#                 A,
#                 Bmat,
#                 Cmat,
#                 self.D.float(),
#                 z=None,
#                 delta_bias=self.dt_proj.bias.float(),
#                 delta_softplus=True,
#                 return_last_state=None,
#             )

#             self._check_tensor("y_after_scan", y)

#             y = torch.cat([y, z], dim=1)
#             y = rearrange(y, "b d l -> b l d")
#             out = self.out_proj(y)
#             self._check_tensor("out", out)
#             return out

#         except RuntimeError as e:
#             # Catch low-level CUDA assert and re-raise with context
#             if "device-side assert" in str(e).lower():
#                 raise RuntimeError(
#                     "[MambaVisionMixer] CUDA device-side assert inside selective_scan_fn "
#                     f"or conv1d.\n"
#                     f"hidden_states shape: {hidden_states.shape}, "
#                     f"d_model={self.d_model}, d_inner={self.d_inner}, "
#                     f"d_state={self.d_state}, seq_len={seqlen}"
#                 ) from e
#             else:
#                 raise


class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
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
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

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

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    # an alternative to forward_corev1
    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn_v1

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        **ssm_kwargs,
    ):
        """
        Vision State Space Module block.

        Args:
            hidden_dim: input/output channel dimension (C).
            drop_path: stochastic depth.
            norm_layer: norm used after the top branch SSM.
            attn_drop_rate: dropout inside SS2D.
            d_state, **ssm_kwargs: forwarded to SS2D.
        """
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be divisible by 2"
        self.hidden_dim = hidden_dim
        self.half_dim = hidden_dim // 2

        # ---------- TOP BRANCH: Linear -> DWConv -> SiLU -> 2D-SSM -> LayerNorm ----------
        self.top_linear = nn.Linear(self.half_dim, self.half_dim)
        self.top_dwconv = nn.Conv2d(
            in_channels=self.half_dim,
            out_channels=self.half_dim,
            kernel_size=3,
            padding=1,
            groups=self.half_dim,   # depthwise
            bias=True,
        )
        self.top_act = nn.SiLU()
        # 2D-SSM (VSSM core)
        self.top_ssm = SS2D(
            d_model=self.half_dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            **ssm_kwargs,
        )
        self.top_norm = norm_layer(self.half_dim)

        # ---------- BOTTOM BRANCH: Linear -> SiLU ----------
        self.bottom_linear = nn.Linear(self.half_dim, self.half_dim)
        self.bottom_act = nn.SiLU()

        # ---------- OUTPUT PROJECTION ----------
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        assert C == self.hidden_dim

        # split channels
        x_top, x_bottom = x.chunk(2, dim=-1)  # each (B, H, W, C/2)

        # ===== TOP BRANCH =====
        t = self.top_linear(x_top)                     # (B, H, W, C/2)

        # DWConv expects (B, C, H, W)
        t_2d = t.permute(0, 3, 1, 2).contiguous()      # (B, C/2, H, W)
        t_2d = self.top_dwconv(t_2d)
        t = t_2d.permute(0, 2, 3, 1).contiguous()      # back to (B, H, W, C/2)

        t = self.top_act(t)
        t = self.top_ssm(t)                            # SS2D keeps (B, H, W, C/2)
        t = self.top_norm(t)

        # ===== BOTTOM BRANCH =====
        b = self.bottom_linear(x_bottom)               # (B, H, W, C/2)
        b = self.bottom_act(b)

        # ===== MERGE & OUTPUT =====
        y = torch.cat([t, b], dim=-1)                  # (B, H, W, C)
        y = self.out_linear(y)                         # (B, H, W, C)

        return y
    
class TokenVSSM(nn.Module):
    """
    Wraps VSSMBlock so it can operate on token sequences (B, N, C).

    Assumes N = H * W and reshapes tokens -> (B, H, W, C) -> VSSM -> back.
    """
    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **ssm_kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.vssm_block = VSSMBlock(
            hidden_dim=dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
            **ssm_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        assert H * W == N, f"TokenVSSM: N={N} must be a perfect square (got H={H},W={W})"
        assert C == self.dim, f"TokenVSSM: channel mismatch, C={C}, dim={self.dim}"

        # (B, N, C) -> (B, H, W, C)
        x_2d = x.view(B, H, W, C).contiguous()

        # Apply VSSM
        y_2d = self.vssm_block(x_2d)   # (B, H, W, C)

        # Back to tokens
        y = y_2d.view(B, N, C).contiguous()
        return y

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
        self.attn = TokenMDTA(dim=dim, num_heads=num_heads, bias=True)

        self.ln2 = nn.LayerNorm(dim)      # for first f-KAN
        self.ffn1 = FKANMLP(dim, mlp_dim)

        # --- Mamba part ---
        print("At VSSM ")
        self.ln3 = nn.LayerNorm(dim)      # for VSSM
        # self.vssm = MambaVisionMixer(
        #     d_model=dim,
        #     d_state=d_state,
        #     d_conv=d_conv,
        #     expand=expand,
        # )
        self.vssm = TokenVSSM(
                    dim=dim,
                    d_state=d_state,
                    drop_path=0.0,          # or pass your drop_path_rate here
                    attn_drop_rate=0.0,     # or whatever you want
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
        print("Initializing SegMamba with Hybrid Encoder along with GSC + VSSM")
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

    
