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

from .kan_fJNB import KAN
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from transformers import AutoTokenizer, AutoModel


# class ClinicalTextEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
#         self.encoder = AutoModel.from_pretrained("medicalai/ClinicalBERT")

#         # freeze text backbone
#         for p in self.encoder.parameters():
#             p.requires_grad = False

#     def forward(self, texts):
#         """
#         texts: list[str] length = B
#         returns: (B, 768)
#         """
#         tokens = self.tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=128,
#             return_tensors="pt"
#         ).to(next(self.encoder.parameters()).device)

#         out = self.encoder(**tokens)
#         # return out.last_hidden_state.mean(dim=1)
#         return out.last_hidden_state   # (B, T, 768)


class ClinicalTextEncoder(nn.Module):
    """
    Frozen ClinicalBERT encoder.
    Returns token-level embeddings: (B, T, 768)
    """

    def __init__(self, model_name="medicalai/ClinicalBERT"):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze backbone
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, texts):
        """
        texts: list[str] length B
        return: (B, T, 768)
        """

        if texts is None:
            return None

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        tokens = {k: v.to(self.encoder.device) for k, v in tokens.items()}
        out = self.encoder(**tokens)

        return out.last_hidden_state

class TGDC(nn.Module):
    """
    Implementation strictly following ViTexNet TGDC module.
    """
    def __init__(self, dim, num_filters=4, kernel_size=3):
        super().__init__()
 
        self.dim = dim
        self.K = num_filters
 
        # Eq (2): Text MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_filters)
        )
 
        # K parallel depthwise 1D conv (Eq 3)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                dim, dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim  # depthwise
            )
            for _ in range(num_filters)
        ])
 
        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(dim))
 
    def global_text_gating(self, T):
        # Eq (1): AdaptiveAvgPool1d
        t_pool = T.mean(dim=1)  # (B, C)
 
        # Eq (2): MLP + Softmax
        weights = self.mlp(t_pool)
        weights = F.softmax(weights, dim=-1)
 
        return weights  # (B, K)
 
    def weighted_fusion(self, x, weights):
        # x: (B, N, C)
        x = x.transpose(1, 2)  # (B, C, N)
 
        fused = 0
        for i, conv in enumerate(self.convs):
            Oi = conv(x)  # (B, C, N)
            wi = weights[:, i].unsqueeze(-1).unsqueeze(-1)
            fused += wi * Oi
 
        fused = fused.transpose(1, 2)  # (B, N, C)
        return fused
 
    def forward(self, V, T):
        """
        V: (B, N, C) visual tokens
        T: (B, L, C) text tokens
        """
 
        # -------- First Pass --------
        weights = self.global_text_gating(T)
        F1 = self.weighted_fusion(V, weights)
        F1 = self.gamma * self.norm(F1)
 
        # -------- Iterative Refinement (Same weights) --------
        F2 = self.weighted_fusion(F1, weights)
        F2 = self.gamma * self.norm(F2)
 
        return F2 + V
    
class TGDCFusion(nn.Module):
    """
    Wrap TGDC so it can fuse text into 3D UNet skip feature maps.
    Input:  x5d = (B, C, D, H, W)
            T   = (B, L, text_dim)
    Output: fused x5d
    """

    def __init__(self, img_dim, text_dim=768, num_filters=4):
        super().__init__()

        # Project text tokens → same dim as image channels
        self.text_proj = nn.Linear(text_dim, img_dim)

        # TGDC works in token space (B, N, C)
        self.tgdc = TGDC(dim=img_dim, num_filters=num_filters)

    def forward(self, x5d, text_tokens):
        """
        x5d:        (B, C, D, H, W)
        text_tokens:(B, L, 768)
        """

        B, C, D, H, W = x5d.shape
        N = D * H * W

        # ---- flatten image → tokens ----
        V = x5d.view(B, C, N).transpose(1, 2)   # (B, N, C)

        # ---- project text tokens ----
        T = self.text_proj(text_tokens)         # (B, L, C)

        # ---- TGDC fusion ----
        fused_tokens = self.tgdc(V, T)          # (B, N, C)

        # ---- reshape back ----
        fused = fused_tokens.transpose(1, 2).view(B, C, D, H, W)

        return fused

import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class HSLCA(nn.Module):
    """
    Hierarchical Summary Linear Cross Attention
    Designed for MIS.
 
    image_tokens: (B, N, C)
    text_tokens:  (B, L, C)
    """
 
    def __init__(self, dim, num_heads=4, num_summary_tokens=4, reduction=4):
        super().__init__()
 
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.K = num_summary_tokens
 
        assert dim % num_heads == 0
 
    #Text summary generator 
        self.summary_proj = nn.Linear(dim, num_summary_tokens)
 
   
        #Cross Attention Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
 
        self.out_proj = nn.Linear(dim, dim)
 
    # GATED FUSION HAI YE, YOU CAN CHNAGE THE nn.linear with KAN
        # self.gate_mlp = nn.Sequential(
        #     nn.Linear(dim, dim // reduction),
        #     nn.ReLU(),
        #     nn.Linear(dim // reduction, dim),
        #     nn.Sigmoid()
        # )
        hidden_gate = dim // reduction
        self.gate_kan = KAN(
            layers_hidden=[dim, hidden_gate, dim],
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            grid_eps=0.02,
            grid_range=[-1, 1],
        )

        self.gate_act = nn.Sigmoid()
        self.norm = nn.LayerNorm(dim)
        self.gate_norm = nn.LayerNorm(dim)
 
    def reshape_heads(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
 
    def merge_heads(self, x):
        B, H, N, D = x.shape
        return x.transpose(1, 2).reshape(B, N, H * D)
 
    def phi(self, x):
        return F.elu(x) + 1  # linear attention trick
 
 
    def forward(self, image_tokens, text_tokens):
 
        B, N, C = image_tokens.shape
        L = text_tokens.shape[1]
 
    #GENERATE TEXT SUMMARY TOKENS 
        # (B, L, C) → (B, L, K)
        summary_scores = self.summary_proj(text_tokens)
 
        # soft clustering across token dimension
        summary_weights = F.softmax(summary_scores, dim=1)
 
        # weighted aggregation
        # (B, K, C)
        text_summary = torch.matmul(
            summary_weights.transpose(1, 2),
            text_tokens
        )
    #LINEAR CROSS ATTENTION HAI YE 
        Q = self.reshape_heads(self.q_proj(image_tokens))
        K = self.reshape_heads(self.k_proj(text_summary))
        V = self.reshape_heads(self.v_proj(text_summary))
 
        Q = self.phi(Q)
        K = self.phi(K)
 
        KV = torch.matmul(K.transpose(-2, -1), V)
        attn_out = torch.matmul(Q, KV)
 
        attn_out = self.merge_heads(attn_out)
        attn_out = self.out_proj(attn_out)
 
    # GATED FUSION HAI YE 
        # gate_input = attn_out.mean(dim=1)
        gate_input = self.gate_norm(attn_out.mean(dim=1))
        # alpha = self.gate_mlp(gate_input).unsqueeze(1)
        # ---- KAN gating ----
        alpha = self.gate_kan(gate_input)   # (B, C)
        alpha = self.gate_act(alpha)        # sigmoid gating
        alpha = alpha.unsqueeze(1)          # (B, 1, C)
 
        fused = image_tokens + alpha * attn_out
 
        return self.norm(fused)
    
class HSLCAFusion(nn.Module):
    """
    Wrap HSLCA so it can fuse text into 3D UNet skip feature maps.

    x5d: (B, C, D, H, W)
    text_tokens: (B, L, text_dim)
    """

    def __init__(
        self,
        img_dim,
        text_dim=768,
        num_heads=4,
        num_summary_tokens=4,
        reduction=4,
    ):
        super().__init__()

        # project text tokens to same dim as image channels
        self.text_proj = nn.Linear(text_dim, img_dim)

        # optional stabilization
        self.norm_img = nn.LayerNorm(img_dim)
        self.norm_txt = nn.LayerNorm(img_dim)

        self.hslca = HSLCA(
            dim=img_dim,
            num_heads=num_heads,
            num_summary_tokens=num_summary_tokens,
            reduction=reduction,
        )

    def forward(self, x5d, text_tokens):
        """
        x5d: (B, C, D, H, W)
        text_tokens: (B, L, 768)
        """

        B, C, D, H, W = x5d.shape
        N = D * H * W

        # ---- flatten image → tokens ----
        V = x5d.view(B, C, N).transpose(1, 2)   # (B, N, C)

        # ---- project text tokens ----
        T = self.text_proj(text_tokens)         # (B, L, C)

        # ---- normalize (important for attention stability) ----
        V = self.norm_img(V)
        T = self.norm_txt(T)

        # ---- HSLCA fusion ----
        fused_tokens = self.hslca(V, T)         # (B, N, C)

        # ---- reshape back ----
        fused = fused_tokens.transpose(1, 2).view(B, C, D, H, W)

        return fused

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


class MambaLayer(nn.Module):
    """
    Tri-oriented Spatial Mamba Block (TSMamba) operating on 3D features.

    Input / output: x5d = (B, C, D, H, W)

    Internally:
      - flatten to tokens (B, N, C)
      - LN -> MDTA -> res
      - LN -> f-KAN -> res
      - LN -> VSSM (MambaVisionMixer) -> res
      - LN -> f-KAN -> res
      - reshape back to (B, C, D, H, W)
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0,
                 d_state=8, d_conv=3, expand=1, num_slices=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        mlp_dim = int(dim * mlp_ratio)

        # LN + MDTA
        print("At Transformer ")
        self.ln1 = nn.LayerNorm(dim)
        self.attn = TokenMDTA(dim=dim, num_heads=num_heads, bias=True)

        # LN + f-KAN (1)
        self.ffn1 = FKANMLP(dim, mlp_dim)

        # LN + VSSM (MambaVisionMixer)
        print("At MambaVisionMixer ")
        self.ln3 = nn.LayerNorm(dim)
        self.vssm = MambaVisionMixer(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # LN + f-KAN (2)
        self.ffn2 = FKANMLP(dim, mlp_dim)

    def forward(self, x5d):
        # x5d: (B, C, D, H, W)
        B, C = x5d.shape[:2]
        spatial = x5d.shape[2:]             # (D, H, W)
        N = spatial[0] * spatial[1] * spatial[2]

        # ----- flatten to tokens -----
        x = x5d.view(B, C, N).transpose(1, 2)   # (B, N, C)
        # print("[MambaLayer] tokens in:", x.shape)

        # LN -> MDTA -> residual
        h = x
        x_ln = self.ln1(x)
        x_attn, _ = self.attn(x_ln)
        x = x_attn + h

        # LN -> f-KAN (1) -> residual
        h = x
        x_ffn1 = self.ffn1(x)
        x = x_ffn1 + h

        # LN -> VSSM -> residual
        h = x
        x_ln3 = self.ln3(x)
        x_vssm = self.vssm(x_ln3)    # (B, N, C)
        x = x_vssm + h

        # LN -> f-KAN (2) -> residual
        h = x
        x_ffn2 = self.ffn2(x)
        x = x_ffn2 + h

        # ----- back to 5D -----
        x_out = x.transpose(1, 2).view(B, C, *spatial)
        # print("[MambaLayer] x_out:", x_out.shape)
        return x_out

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


    def forward(self, x5d):
        # x5d: (B, C, D, H, W)
        B, C = x5d.shape[:2]
        D, H, W = x5d.shape[2:]
        N = D * H * W
        # print(f"[TMB] x5d in:         {x5d.shape}")

        # ===== flatten to tokens =====
        x = x5d.view(B, C, N).transpose(1, 2)   # (B, N, C)
        x_in = x                                # original input tokens
        # print(f"[TMB] tokens x_in:    {x_in.shape}")   # (B, N, C)

        # ================= TRANSFORMER PART =================
        # 1) LN -> MDTA -> add residual (orig input)
        t = self.ln1(x_in)
        # print(f"[TMB] after ln1:      {t.shape}")
        t, _ = self.attn(t)                     # (B, N, C)
        # print(f"[TMB] after attn:     {t.shape}")
        t = x_in + t                            # attn_residual
        # print(f"[TMB] after attn res: {t.shape}")

        # 2) LN -> f-KAN -> add residual (orig input)
        u = self.ln2(t)
        # print(f"[TMB] after ln2:      {u.shape}")
        u = self.ffn1(u)                        # (B, N, C)
        # print(f"[TMB] after fKAN1:    {u.shape}")
        u = u + t                            # f-KAN residual
        # print(f"[TMB] after fKAN1 res:{u.shape}")
        x_tr = x_in + u                         # transformer output
        # print(f"[TMB] x_tr:           {x_tr.shape}")

        # ================== MAMBA PART =====================
        # 3) LN -> VSSM -> add residual (transformer output)
        m = self.ln3(x_tr)
        # print(f"[TMB] after ln3:      {m.shape}")
        m = self.vssm(m)                        # (B, N, C)
        # print(f"[TMB] after VSSM:     {m.shape}")
        m = x_tr + m                            # mamba_residual
        # print(f"[TMB] after VSSM res: {m.shape}")

        # 4) LN -> f-KAN -> add residual (transformer output)
        n = self.ln4(m)
        # print(f"[TMB] after ln4:      {n.shape}")
        n = self.ffn2(n)                        # (B, N, C)
        # print(f"[TMB] after fKAN2:    {n.shape}")
        n = n + m                            # f-KAN residual
        # print(f"[TMB] after fKAN2 res:{n.shape}")
        x_out_tokens = x_tr + n                 # final output tokens
        # print(f"[TMB] x_out_tokens:   {x_out_tokens.shape}")

        # ===== back to 5D =====
        x_out = x_out_tokens.transpose(1, 2).view(B, C, D, H, W)
        # print(f"[TMB] x_out 5D:       {x_out.shape}")
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

        # print(f"[GSC] in:   {x.shape}")
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
        
        # print(f"[GSC] out:  {x.shape}")
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
        # print(f"[MambaEncoder] input:       {x.shape}")

        for i in range(4):
            # 3D downsampling (unchanged)
            x = self.downsample_layers[i](x)
            # print(f"[MambaEncoder] after downsample[{i}]: {x.shape}")

            x = self.gscs[i](x)
            # print(f"[MambaEncoder] after GSC[{i}]:        {x.shape}")

            # Token pipeline block(s) on flattened tokens
            x = self.stages[i](x)
            # print(f"[MambaEncoder] after stage[{i}]:      {x.shape}")

            if i in self.out_indices:
                outs.append(x)
                # print(f"[MambaEncoder] -> outs[{len(outs)-1}] shape: {x.shape}")

        return tuple(outs)

    def forward(self, x):
        return self.forward_features(x)


class FinalKANRefine3D(nn.Module):
    """
    KAN-based refinement between final decoder and logits head.

    Input / output: [B, C, D, H, W]
    """
    def __init__(self, in_channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        print("Initializing final KAN refinement block")
        mlp_dim = int(in_channels * mlp_ratio)
        self.kan_mlp = FKANMLP(dim=in_channels, mlp_dim=mlp_dim)

    def forward(self, x5d: torch.Tensor) -> torch.Tensor:
        # x5d: [B, C, D, H, W]
        B, C, D, H, W = x5d.shape
        N = D * H * W

        # [B, C, D, H, W] -> [B, N, C]
        x = x5d.view(B, C, N).transpose(1, 2)   # [B, N, C]

        # KAN token MLP
        x = self.kan_mlp(x)                     # [B, N, C]

        # back to [B, C, D, H, W]
        x = x.transpose(1, 2).view(B, C, D, H, W)
        return x

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
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.deep_supervision = deep_supervision

        # print("Initializing SegMamba")
        print("Initializing SegMamba with Hybrid Encoder - GSC + MDTA + MambaVisionMixer + KAN-Refine with Deep Supervision")
        print("With Text Infusion via Hierarchical Summary Linear Cross Attention ")
        self.spatial_dims = spatial_dims
        # ---- TEXT ENCODER ----
        self.text_encoder = ClinicalTextEncoder()

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

        # ---- TEXT → SKIP FiLM modules ----
        # self.skip_film1 = SkipFiLM(self.feat_size[0])   # enc1 skip
        # self.skip_film2 = SkipFiLM(self.feat_size[1])   # enc2 skip
        # self.skip_film3 = SkipFiLM(self.feat_size[2])   # enc3 skip
        # self.skip_film4 = SkipFiLM(self.feat_size[3])   # enc4 skip

        # ---- TEXT → SKIP Cross Attention Fusion ----
        # self.cross_attn1 = CrossAttentionFusion(img_dim=self.feat_size[0])
        # self.cross_attn2 = CrossAttentionFusion(img_dim=self.feat_size[1])
        # self.cross_attn3 = CrossAttentionFusion(img_dim=self.feat_size[2])
        # self.cross_attn4 = CrossAttentionFusion(img_dim=self.feat_size[3])

        # self.tgdc1 = TGDCFusion(img_dim=self.feat_size[0])
        # self.tgdc2 = TGDCFusion(img_dim=self.feat_size[1])
        # self.tgdc3 = TGDCFusion(img_dim=self.feat_size[2])
        # self.tgdc4 = TGDCFusion(img_dim=self.feat_size[3])

        self.hslca1 = HSLCAFusion(img_dim=self.feat_size[0])
        self.hslca2 = HSLCAFusion(img_dim=self.feat_size[1])
        self.hslca3 = HSLCAFusion(img_dim=self.feat_size[2])
        self.hslca4 = HSLCAFusion(img_dim=self.feat_size[3])
        self.hslca_hidden = HSLCAFusion(img_dim=self.hidden_size)


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

        self.final_refine = FinalKANRefine3D(
                                in_channels=self.feat_size[0],  # 48
                                mlp_ratio=4.0,                  # tweakable
                            )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

        # ============ DEEP SUPERVISION HEADS ============
        if self.deep_supervision:
            # after decoder5 -> dec3: channels = feat_size[3]
            print("Using deep supervision heads")
            self.ds_head3 = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[3],
                out_channels=self.out_chans,
            )
            # after decoder4 -> dec2: channels = feat_size[2]
            self.ds_head2 = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[2],
                out_channels=self.out_chans,
            )
            # after decoder3 -> dec1: channels = feat_size[1]
            self.ds_head1 = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[1],
                out_channels=self.out_chans,
            )

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    
    # def forward(self, x_in):
    def forward(self, x_in, text=None):

        """
        x_in: [B, C, H, W] or [B, C, D, H, W]
        """
        squeeze_depth = False
        # ---- Text embedding ----
        # text_emb = self.text_encoder(text)   # (B, 768)
        text_tokens = self.text_encoder(text)   # (B, T, 768)

        # print(f"[SegMamba] x_in raw:        {x_in.shape}")

        # ---- Input checks ----
        if x_in.dim() not in (4, 5):
            raise RuntimeError(
                f"[SegMamba] Expected 4D or 5D input, got {x_in.dim()}D with shape {x_in.shape}"
            )

        if x_in.dim() == 4:
            x_in = x_in.unsqueeze(2)   # [B, C, 1, H, W]
            squeeze_depth = True
            # print(f"[SegMamba] x_in unsqueezed: {x_in.shape}")

        if x_in.size(1) != self.in_chans:
            raise RuntimeError(
                f"[SegMamba] Channel mismatch: got {x_in.size(1)} channels, "
                f"expected {self.in_chans}"
            )

        # --- Encoder path with Mamba features ---
        outs = self.vit(x_in)        # tuple of 4 feature maps
        for i, o in enumerate(outs):
            # print(f"[SegMamba] vit outs[{i}]:  {o.shape}")
            if o is None:
                raise RuntimeError(f"[SegMamba] vit outs[{i}] is None")

        enc1 = self.encoder1(x_in)
        # enc1 = self.cross_attn1(enc1, text_tokens)
        enc1 = self.hslca1(enc1, text_tokens)
        # print(f"[SegMamba] enc1:           {enc1.shape}")

        x2 = outs[0]
        enc2 = self.encoder2(x2)
        # print(f"[SegMamba] enc2:           {enc2.shape}")
        # enc2 = self.cross_attn2(enc2, text_tokens)
        enc2 = self.hslca2(enc2, text_tokens)

        x3 = outs[1]
        enc3 = self.encoder3(x3)
        # print(f"[SegMamba] enc:           {enc3.shape}")
        # enc3 = self.cross_attn3(enc3, text_tokens)
        enc3 = self.hslca3(enc3, text_tokens)

        x4 = outs[2]
        enc4 = self.encoder4(x4)
        # print(f"[SegMamba] enc4:           {enc4.shape}")
        # enc4 = self.cross_attn4(enc4, text_tokens)
        enc4 = self.hslca4(enc4, text_tokens)

        enc_hidden = self.encoder5(outs[3])
        enc_hidden = self.hslca_hidden(enc_hidden, text_tokens)

        # print(f"[SegMamba] enc_hidden:     {enc_hidden.shape}")

        # --- Decoder path ---
        dec3 = self.decoder5(enc_hidden, enc4)
        # print(f"[SegMamba] dec3:           {dec3.shape}"

        dec2 = self.decoder4(dec3, enc3)
        # print(f"[SegMamba] dec2:           {dec2.shape}")

        dec1 = self.decoder3(dec2, enc2)
        # print(f"[SegMamba] dec1:           {dec1.shape}")

        dec0 = self.decoder2(dec1, enc1)
        # print(f"[SegMamba] dec0:           {dec0.shape}")

        out = self.decoder1(dec0)
        # print(f"[SegMamba] decoder1_out:   {out.shape}")

         # === KAN refinement step ===
        out = self.final_refine(out)
        # print(f"[SegMamba] final_refine_out: {out.shape}")

        # ===== main prediction =====
        out_main = self.out(out)                  # [B, out_chans, D, H, W]
        # print(f"[SegMamba] out_main logits:   {out_main.shape}")
        
        # ===== deep supervision predictions =====
        ds1_up = ds2_up = ds3_up = None
        if self.deep_supervision:
            # side-head predictions at their native scale
            ds3 = self.ds_head3(dec3)            # [B, out_chans, D, H/8,  W/8] (example)
            ds2 = self.ds_head2(dec2)            # [B, out_chans, D, H/4,  W/4]
            ds1 = self.ds_head1(dec1)            # [B, out_chans, D, H/2,  W/2]

            # print(f"[SegMamba] ds3 raw:        {ds3.shape}")
            # print(f"[SegMamba] ds2 raw:        {ds2.shape}")
            # print(f"[SegMamba] ds1 raw:        {ds1.shape}")

            # upsample all to match main output resolution
            target_size = out_main.shape[2:]     # (D, H, W)
            # print(f"[SegMamba] target_size for upsample: {target_size}")
            ds3_up = F.interpolate(
                ds3, size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            ds2_up = F.interpolate(
                ds2, size=target_size,
                mode="trilinear",
                align_corners=False,
            )
            ds1_up = F.interpolate(
                ds1, size=target_size,
                mode="trilinear",
                align_corners=False,
            )

            # print(f"[SegMamba] ds3_up:         {ds3_up.shape}")
            # print(f"[SegMamba] ds2_up:         {ds2_up.shape}")
            # print(f"[SegMamba] ds1_up:         {ds1_up.shape}")

        # ===== squeeze fake depth dim (for 2D use) =====
        if squeeze_depth:
            out_main = out_main.squeeze(2)       # [B, out_chans, H, W]
            # print(f"[SegMamba] out_main 2D:     {out_main.shape}")

            if self.deep_supervision:
                ds3_up = ds3_up.squeeze(2)
                ds2_up = ds2_up.squeeze(2)
                ds1_up = ds1_up.squeeze(2)
                # print(f"[SegMamba] ds3_up 2D:     {ds3_up.shape}")
                # print(f"[SegMamba] ds2_up 2D:     {ds2_up.shape}")
                # print(f"[SegMamba] ds1_up 2D:     {ds1_up.shape}")

        # UNCOMMENT WHEN TRAIN AND TESTING WITH DEEP SUPERVISION
        
        # ===== return =====
        if self.deep_supervision:
            # main output first, aux outputs after
            return out_main, ds1_up, ds2_up, ds3_up
        else:
            return out_main
        
        # return out_main

  