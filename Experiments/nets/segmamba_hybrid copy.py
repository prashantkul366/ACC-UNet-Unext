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
from . import vit_seg_configs as configs
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F 
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
from .kan_fJNB import KAN
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
import math
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

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v2",
                # nslices=num_slices,
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        
        return out
    
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
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        # stem = nn.Sequential(
        #       nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        #       )
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
        # for i in range(3):
        #     downsample_layer = nn.Sequential(
        #         # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
        #         nn.InstanceNorm3d(dims[i]),
        #         nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
        #     )
        #     self.downsample_layers.append(downsample_layer)

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

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

# class SegMamba(nn.Module):
#     def __init__(
#         self,
#         in_chans=1,
#         out_chans=13,
#         depths=[2, 2, 2, 2],
#         feat_size=[48, 96, 192, 384],
#         drop_path_rate=0,
#         layer_scale_init_value=1e-6,
#         hidden_size: int = 768,
#         norm_name = "instance",
#         conv_block: bool = True,
#         res_block: bool = True,
#         spatial_dims=3,
#     ) -> None:
#         super().__init__()


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


class KANMLP(nn.Module):
    """
    Drop-in replacement for Mlp that uses KAN.
    Expects (B, N, D) in and out. Internally flattens tokens to (B*N, D).
    """
    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        mlp_dim = config.transformer["mlp_dim"]

        # Read KAN hyperparams from config with sensible defaults
        self.grid_size    = getattr(config, "kan_grid_size", 5)
        self.spline_order = getattr(config, "kan_spline_order", 3)
        self.scale_noise  = getattr(config, "kan_scale_noise", 0.1)
        self.scale_base   = getattr(config, "kan_scale_base", 1.0)
        self.scale_spline = getattr(config, "kan_scale_spline", 1.0)
        self.grid_eps     = getattr(config, "kan_grid_eps", 0.02)
        self.grid_range   = getattr(config, "kan_grid_range", [-1, 1])

        # Two-layer MLP path: hidden -> mlp_dim -> hidden, but using KAN
        self.kan = KAN(
            layers_hidden=[hidden, mlp_dim, hidden],
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            scale_noise=self.scale_noise,
            scale_base=self.scale_base,
            scale_spline=self.scale_spline,
            grid_eps=self.grid_eps,
            grid_range=self.grid_range,
        )

        # Keep the same dropout semantics as the original Mlp
        self.dropout = Dropout(config.transformer["dropout_rate"])

        # Optional: tiny LayerNorm to stabilize ranges before KAN (KAN grid default is [-1,1])
        # LayerNorm keeps zero mean/unit variance, which is friendly to KAN splines.
        self.pre_norm = LayerNorm(hidden, eps=1e-6)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        x = self.pre_norm(x)

        x_flat = x.reshape(B * N, D)   # (BN, D)
        y_flat = self.kan(x_flat)      # (BN, D)
        y = y_flat.reshape(B, N, D)    # (B, N, D)

        # Match original MLP’s dropout-after-each-linear feel
        y = self.dropout(y)
        return y

    def regularization_loss(self, reg_act=1.0, reg_ent=1.0):
        # Expose KAN’s regularizer so your training loop can add it
        return self.kan.regularization_loss(reg_act, reg_ent)

class T_Block(nn.Module):
    def __init__(self, config, vis):
        super(T_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)

        # print("Transformer block with MDTA + KAN MLP initiated")
        # ############################################################
        # # self.ffn = Mlp(config)
        # self.ffn = KANMLP(config)
        # ############################################################
        
        num_heads = config.transformer.get("num_heads", 4)
        self.attn = TokenMDTA(dim=self.hidden_size, num_heads=num_heads, bias=True)
        self.ffn = KANMLP(config)
        

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights
    

    
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

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class M_Block(nn.Module):
    def __init__(self, config, vis):
        super(M_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.vis = vis

        # print("Block with MambaVisionMixer + fJNB KAN instead of MLP initiated")
        
        self.attn = MambaVisionMixer(
            d_model=config.hidden_size,
            d_state=8,
            d_conv=3,
            expand=1,
        )
        self.ffn = KANMLP(config)


    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        weights = None
        return x, weights

    
class TSMambaAdapter(nn.Module):
    """
    Drop-in replacement for Transformer when using a 2D Mamba encoder.
    Returns (tokens, attn_weights, features) to feed DecoderCup.
    """
    def __init__(self, config, img_size, vis=False):
        super().__init__()
        self.config = config
        self.vis = vis
        self.hidden_size = config.hidden_size

        # 4 encoder stages, each stage = Transformer block + Mamba block
        self.num_stages = 4
        self.stages = nn.ModuleList([
            nn.ModuleList([
                T_Block(config, vis),
                M_Block(config, vis),
            ]) for _ in range(self.num_stages)
        ])



    def forward(self, x):
        # x: (B, C, 224, 224)
        outs = self.backbone(x)   # (f0, f1, f2, f3)

        f0, f1, f2, f3 = outs     # shapes:
                                  # f0: (B, 48, 112, 112)
                                  # f1: (B, 96, 56, 56)
                                  # f2: (B,192, 28, 28)
                                  # f3: (B,384, 14, 14)

        # bottleneck to tokens: (B, 196, hidden_size)
        B, C, H, W = f3.shape
        tokens = f3.flatten(2).transpose(1, 2)  # (B, H*W, C) = (B, 196, 384)

        # 3 skip connections: H/8, H/4, H/2
        features = [
            f2,  # (B,192, 28, 28)
            f1,  # (B, 96, 56, 56)
            f0,  # (B, 48,112,112)
        ]

        attn_weights = None
        return tokens, attn_weights, features

class TSMambaUNETREncoder(nn.Module):
    def __init__(self, tsmamba, feat_size=[48, 96, 192, 384]):
        super().__init__()
        self.core = tsmamba      # this is your TSMambaAdapter (T_Block + M_Block)
        self.feat_size = feat_size

    def forward(self, x):
        """
        x: [B, C, 1, H, W] or [B, C, H, W]

        Return:
        (
          f0_3d: [B, 48, 1, H/2,  W/2],
          f1_3d: [B, 96, 1, H/4,  W/4],
          f2_3d: [B,192, 1, H/8,  W/8],
          f3_3d: [B,384, 1, H/16, W/16],
        )
        """
        # --- 1) Go to 2D for your encoder ---
        if x.dim() == 5:
            x2d = x.squeeze(2)         # [B, C, H, W]
            print("[TSMambaUNETREncoder] input 5D:", x.shape)
        else:
            x2d = x                    # already 2D

        print("[TSMambaUNETREncoder] x2d:", x2d.shape)

        tokens, _, features = self.core(x2d)
        print("[TSMambaUNETREncoder] tokens:", tokens.shape)
        for i, f in enumerate(features):
            print(f"[TSMambaUNETREncoder] features[{i}] 2D:", f.shape)

        # features as you defined earlier: [f2, f1, f0]
        f2, f1, f0 = features

        # reconstruct f3 from tokens
        B, N, C3 = tokens.shape       # e.g. N=14*14, C3=384
        h3 = w3 = int(N ** 0.5)
        f3 = tokens.transpose(1, 2).reshape(B, C3, h3, w3)
        print("[TSMambaUNETREncoder] f3 2D:", f3.shape)


        # reorder to match SegMamba expectation: H/2, H/4, H/8, H/16
        f0_3d = f0.unsqueeze(2)       # [B, 48, 1, 112, 112]
        f1_3d = f1.unsqueeze(2)       # [B, 96, 1, 56,  56 ]
        f2_3d = f2.unsqueeze(2)       # [B,192, 1, 28,  28 ]
        f3_3d = f3.unsqueeze(2)       # [B,384, 1, 14,  14 ]
        
        print("[TSMambaUNETREncoder] f0_3d:", f0_3d.shape)
        print("[TSMambaUNETREncoder] f1_3d:", f1_3d.shape)
        print("[TSMambaUNETREncoder] f2_3d:", f2_3d.shape)
        print("[TSMambaUNETREncoder] f3_3d:", f3_3d.shape)

        return (f0_3d, f1_3d, f2_3d, f3_3d)

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
        config = configs.get_r50_b16_config()
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.config = config

        print("Initializing SegMamba with Hybrid Encoder")
        self.spatial_dims = spatial_dims
        # self.vit = MambaEncoder(in_chans, 
        #                         depths=depths,
        #                         dims=feat_size,
        #                         drop_path_rate=drop_path_rate,
        #                         layer_scale_init_value=layer_scale_init_value,
        #                       )
        self.vit_core = TSMambaAdapter(self.config, img_size=224, vis=False)
        self.vit = TSMambaUNETREncoder(self.vit_core, feat_size=self.feat_size)

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

    def forward(self, x_in):
        """
        x_in comes from your ACC-UNet pipeline as [B, C, H, W].

        We:
        - unsqueeze a fake depth dim -> [B, C, 1, H, W]
        - run the 3D Mamba encoder + UNETR blocks
        - squeeze depth back -> [B, out_chans, H, W]
        """
        squeeze_depth = False
        print("[SegMamba] x_in:", x_in.shape) 

        if x_in.dim() == 4:
            # [B, C, H, W] -> [B, C, 1, H, W]
            x_in = x_in.unsqueeze(2)
            squeeze_depth = True
        print("[SegMamba] x_in after unsqueeze:", x_in.shape)

        # --- Encoder path with Mamba features as in your original code ---
        outs = self.vit(x_in)        # tuple of 4 feature maps
        for i, f in enumerate(outs):
            print(f"[SegMamba] vit outs[{i}]:", f.shape)

        enc1 = self.encoder1(x_in)   # skip at full res
        print("[SegMamba] enc1:", enc1.shape)

        x2 = outs[0]
        enc2 = self.encoder2(x2)
        print("[SegMamba] enc2:", enc2.shape)

        x3 = outs[1]
        enc3 = self.encoder3(x3)
        print("[SegMamba] enc3:", enc3.shape)

        x4 = outs[2]
        enc4 = self.encoder4(x4)
        print("[SegMamba] enc4:", enc4.shape)

        enc_hidden = self.encoder5(outs[3])
        print("[SegMamba] enc_hidden:", enc_hidden.shape)

        # --- Decoder path ---
        dec3 = self.decoder5(enc_hidden, enc4)
        print("[SegMamba] dec3:", dec3.shape)

        dec2 = self.decoder4(dec3, enc3)
        print("[SegMamba] dec2:", dec2.shape)

        dec1 = self.decoder3(dec2, enc2)
        print("[SegMamba] dec1:", dec1.shape)

        dec0 = self.decoder2(dec1, enc1)
        print("[SegMamba] dec0:", dec0.shape)

        out = self.decoder1(dec0)     # [B, C, D, H, W]
        print("[SegMamba] out before final conv:", out.shape)

        out = self.out(out)           # [B, out_chans, D, H, W]
        print("[SegMamba] out after final conv:", out.shape)

        if squeeze_depth:
            out = out.squeeze(2)      # [B, out_chans, H, W]
            print("[SegMamba] out after squeeze depth:", out.shape)

        return out

    

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
    'MAMBA-VSS': configs.get_mamba_vss_config(),
    'MOBILE-MAMBA': configs.get_mobilemamba_config(),
    'KAT' : configs.get_kat_b16_config(),
    'TS-MAMBA': configs.get_tsmamba_config(),
}