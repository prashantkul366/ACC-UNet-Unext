import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
# from utils import *
__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
import pdb

# from TinyU_Net import CMRF
from nets.archs.TinyU_Net import CMRF

from nets.archs.wavelet_pool2d import StaticWaveletPool2d
import pywt

# from Topformer import InjectionMultiSumCBR  # SIM module
from nets.archs.Topformer import InjectionMultiSumCBR


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
#     def shift(x, dim):
#         x = F.pad(x, "constant", 0)
#         x = torch.chunk(x, shift_size, 1)
#         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#         x = torch.cat(x, 1)
#         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class UNext_CMRF_GS_Wavelet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self, n_channels=3, n_classes=1,  deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        print("UNext CMRF Encoders + Global Semantic + Sematic Injection Module w Wavelet Pooling Initiated")
        # self.encoder1 = nn.Conv2d(n_channels, 16, 3, stride=1, padding=1)  
        # self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        # self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.encoder1 = CMRF(n_channels, 16)
        self.encoder2 = CMRF(16, 32)
        self.encoder3 = CMRF(32, 128)

        self.pool1 = StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'), scales=1)
        self.pool2 = StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'), scales=1)
        self.pool3 = StaticWaveletPool2d(wavelet=pywt.Wavelet('haar'), scales=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)
        
        self.norm3 = norm_layer(embed_dims[1])
        # self.norm4 = norm_layer(embed_dims[2])
        self.norm4_main = norm_layer(embed_dims[2])  # for out_main_tokens
        self.norm4_gs   = norm_layer(embed_dims[2])  # for g_tokens


        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])
        
        # ---- Global semantics (pool all encoders -> concat -> project) ----
        self.gs_size = img_size // 32                   # 224 -> 7
        self.gs_pool = nn.AdaptiveAvgPool2d((self.gs_size, self.gs_size))

        # t1:16, t2:32, t3:128  => concat 176ch -> project to 256 (bottleneck dim)
        # self.g_in_proj = nn.Conv2d(16 + 32 + 128, 256, kernel_size=1, bias=False)
        self.g_in_proj = nn.Conv2d(16 + 32 + 128 + 160, 256, kernel_size=1, bias=False)
        self.g_in_bn   = nn.BatchNorm2d(256)

        # After bottleneck, split into per-scale chunks (160,128,32,16) for SIMs
        self.g_split_proj = nn.Conv2d(256, 160 + 128 + 32 + 16, kernel_size=1, bias=True)

        # ---- Semantic Injection Modules (TopFormer SIMs) ----
        self.sim4 = InjectionMultiSumCBR(inp=160, oup=160)  # inject at the t4 skip
        self.sim3 = InjectionMultiSumCBR(inp=128, oup=128)  # inject at the t3 skip
        self.sim2 = InjectionMultiSumCBR(inp=32,  oup=32)   # inject at the t2 skip
        self.sim1 = InjectionMultiSumCBR(inp=16,  oup=16)   # inject at the t1 skip


        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(160, 128, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(128, 32, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)
        
        self.final = nn.Conv2d(16, n_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        # --- encoders ---
        # out = F.relu(self.pool1(self.encoder1(x), 2, 2));  t1 = out          # (B,16,112,112)
        # out = F.relu(self.pool2(self.encoder2(out), 2, 2)); t2 = out          # (B,32,56,56)
        # out = F.relu(self.pool3(self.encoder3(out), 2, 2)); t3 = out          # (B,128,28,28)

        out = F.relu(self.pool1(self.encoder1(x)));        t1 = out   # (B,16,112,112)
        out = F.relu(self.pool2(self.encoder2(out)));      t2 = out   # (B,32,56,56)
        out = F.relu(self.pool3(self.encoder3(out)));      t3 = out   # (B,128,28,28)


        # --- stage 4 (tokenized MLP) to get t4 ---
        out, H, W = self.patch_embed3(out)
        for blk in self.block1:
            out = blk(out, H, W)
        out = self.norm3(out)
        t4 = out.reshape(out.shape[0], H, W, -1).permute(0,3,1,2).contiguous()  # (B,160,14,14)

        # ===== MAIN BOTTLENECK (do NOT use block2 here) =====
        out_main_tokens, H4, W4 = self.patch_embed4(t4)          # tokens for main path
        # out_main = self.norm4(out_main_tokens)                   # (B,49,256)
        out_main = self.norm4_main(out_main_tokens)
        out_main = out_main.reshape(B, H4, W4, -1).permute(0,3,1,2).contiguous()  # (B,256,7,7)

        # ===== GLOBAL SEMANTICS: pool+concat t1,t2,t3,t4 =====
        g_cat = torch.cat([
            self.gs_pool(t1),    # (B,16,7,7)
            self.gs_pool(t2),    # (B,32,7,7)
            self.gs_pool(t3),    # (B,128,7,7)
            self.gs_pool(t4),    # (B,160,7,7)
        ], dim=1)                # (B,336,7,7)

        g = self.g_in_bn(self.g_in_proj(g_cat))                   # (B,256,7,7)

        # ---- pass concatenated global semantic through block2 ONLY ----
        g_tokens = g.flatten(2).transpose(1, 2)                   # (B,49,256)
        
        for blk in self.block2:
            g_tokens = blk(g_tokens, self.gs_size, self.gs_size)
        # g_tokens = self.norm4(g_tokens)
        g_tokens = self.norm4_gs(g_tokens)
        g = g_tokens.reshape(B, self.gs_size, self.gs_size, 256).permute(0,3,1,2).contiguous()  # (B,256,7,7)

        # split per scale to drive SIMs for t4,t3,t2,t1
        g = self.g_split_proj(g)                                  # (B,336,7,7)
        g160, g128, g32, g16 = torch.split(g, [160, 128, 32, 16], dim=1)

        ### Stage 4

        # out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out_main)), scale_factor=(2, 2), mode='bilinear'))
        
        if t4.shape[2:] != out.shape[2:]:
           t4 = F.interpolate(t4, size=out.shape[2:], mode='bilinear', align_corners=True)

        t4_aug = self.sim4(t4, F.interpolate(g160, size=out.shape[2:], mode='bilinear', align_corners=False))
        out = out + t4_aug

        # out = torch.add(out,t4)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W) 

        ### Stage 3
        
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        if t3.shape[2:] != out.shape[2:]:
           t3 = F.interpolate(t3, size=out.shape[2:], mode='bilinear', align_corners=True)

        t3_aug = self.sim3(t3, F.interpolate(g128, size=out.shape[2:], mode='bilinear', align_corners=False))
        out = out + t3_aug
          
        # out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        if t2.shape[2:] != out.shape[2:]:
           t2 = F.interpolate(t2, size=out.shape[2:], mode='bilinear', align_corners=True)
        
        t2_aug = self.sim2(t2, F.interpolate(g32, size=out.shape[2:], mode='bilinear', align_corners=False))
        out = out + t2_aug      
        # out = torch.add(out,t2)


        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        if t1.shape[2:] != out.shape[2:]:
          t1 = F.interpolate(t1, size=out.shape[2:], mode='bilinear', align_corners=True)
        
        t1_aug = self.sim1(t1, F.interpolate(g16, size=out.shape[2:], mode='bilinear', align_corners=False))
        out = out + t1_aug
        # out = torch.add(out,t1)


        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        # return self.final(out)
        out = self.final(out)
        if out.shape[1] == 1:
            out = torch.sigmoid(out)  # For binary segmentation
        return out


# class UNext_S(nn.Module):

#     ## Conv 3 + MLP 2 + shifted MLP w less parameters
    
#     def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[32, 64, 128, 512],
#                  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
#                  depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
#         super().__init__()
        
#         self.encoder1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)  
#         self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)  
#         self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

#         self.ebn1 = nn.BatchNorm2d(8)
#         self.ebn2 = nn.BatchNorm2d(16)
#         self.ebn3 = nn.BatchNorm2d(32)
        
#         self.norm3 = norm_layer(embed_dims[1])
#         self.norm4 = norm_layer(embed_dims[2])

#         self.dnorm3 = norm_layer(64)
#         self.dnorm4 = norm_layer(32)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

#         self.block1 = nn.ModuleList([shiftedBlock(
#             dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[0])])

#         self.block2 = nn.ModuleList([shiftedBlock(
#             dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[0])])

#         self.dblock1 = nn.ModuleList([shiftedBlock(
#             dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[0])])

#         self.dblock2 = nn.ModuleList([shiftedBlock(
#             dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
#             sr_ratio=sr_ratios[0])])

#         self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
#                                               embed_dim=embed_dims[1])
#         self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
#                                               embed_dim=embed_dims[2])

#         self.decoder1 = nn.Conv2d(128, 64, 3, stride=1,padding=1)  
#         self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  
#         self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1) 
#         self.decoder4 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
#         self.decoder5 =   nn.Conv2d(8, 8, 3, stride=1, padding=1)

#         self.dbn1 = nn.BatchNorm2d(64)
#         self.dbn2 = nn.BatchNorm2d(32)
#         self.dbn3 = nn.BatchNorm2d(16)
#         self.dbn4 = nn.BatchNorm2d(8)
        
#         self.final = nn.Conv2d(8, num_classes, kernel_size=1)

#         self.soft = nn.Softmax(dim =1)

#     def forward(self, x):
        
#         B = x.shape[0]
#         ### Encoder
#         ### Conv Stage

#         ### Stage 1
#         out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
#         t1 = out
#         ### Stage 2
#         out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
#         t2 = out
#         ### Stage 3
#         out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
#         t3 = out

#         ### Tokenized MLP Stage
#         ### Stage 4

#         out,H,W = self.patch_embed3(out)
#         for i, blk in enumerate(self.block1):
#             out = blk(out, H, W)
#         out = self.norm3(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         t4 = out

#         ### Bottleneck

#         out ,H,W= self.patch_embed4(out)
#         for i, blk in enumerate(self.block2):
#             out = blk(out, H, W)
#         out = self.norm4(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         ### Stage 4

#         out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        
#         out = torch.add(out,t4)                     # FIRST SKIP
#         _,_,H,W = out.shape
#         out = out.flatten(2).transpose(1,2)
#         for i, blk in enumerate(self.dblock1):
#             out = blk(out, H, W)

#         ### Stage 3
        
#         out = self.dnorm3(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
#         out = torch.add(out,t3)                 # SECOND SKIP
#         _,_,H,W = out.shape
#         out = out.flatten(2).transpose(1,2)
        
#         for i, blk in enumerate(self.dblock2):
#             out = blk(out, H, W)

#         out = self.dnorm4(out)
#         out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

#         out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
#         out = torch.add(out,t2)                     # THIRD SKIP
#         out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
#         out = torch.add(out,t1)                     # FOURTH SKIP
#         out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

#         return self.final(out)

if __name__ == '__main__':
    # Sanity check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNext_CMRF_GS_Wavelet(num_classes=1, input_channels=3)
    model.eval()

    # Dummy input: B x C x H x W
    dummy_input = torch.randn(1, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"✅ Forward pass successful! Output shape: {output.shape}")



#EOF