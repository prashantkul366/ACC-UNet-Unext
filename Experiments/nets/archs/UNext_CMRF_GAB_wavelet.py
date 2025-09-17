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

# from ege_unet import group_aggregation_bridge
from nets.archs.ege_unet import group_aggregation_bridge


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

class UNext_CMRF_GAB_Wavelet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    def __init__(self, n_channels=3, n_classes=1,  deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1],use_gab=True, gt_ds=True, **kwargs):
        super().__init__()
        
        self.use_gab = use_gab
        self.gt_ds   = gt_ds
        print("UNext CMRF Encoders + GAB w Wavelet Pooling Initiated")
        print("GT_DS:", self.gt_ds)
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
        self.norm4 = norm_layer(embed_dims[2])

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
        
        # === GAB bridges (dims must be divisible by 4 on xl side) ===
        if self.use_gab:
            self.GAB4 = group_aggregation_bridge(dim_xh=256, dim_xl=160)  # bottleneck -> t4
            self.GAB3 = group_aggregation_bridge(dim_xh=160, dim_xl=128)  # fused l4 -> t3
            self.GAB2 = group_aggregation_bridge(dim_xh=128, dim_xl=32)   # fused l3 -> t2
            self.GAB1 = group_aggregation_bridge(dim_xh=32,  dim_xl=16)   # fused l2 -> t1

        # === Deep supervision mask heads (1 channel each) ===
        if self.use_gab and self.gt_ds:
            self.gt_conv4 = nn.Conv2d(160, 1, 1)  # mask for level-4 GAB (before adding t4)
            self.gt_conv3 = nn.Conv2d(128, 1, 1)  # for level-3
            self.gt_conv2 = nn.Conv2d(32,  1, 1)  # for level-2
            self.gt_conv1 = nn.Conv2d(16,  1, 1)  # for level-1

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

    # def forward(self, x):
        
    #     B = x.shape[0]
    #     ### Encoder
    #     ### Conv Stage

    #     ### Stage 1
    #     # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
    #     # t1 = out
    #     # ### Stage 2
    #     # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
    #     # t2 = out
    #     # ### Stage 3
    #     # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
    #     # t3 = out

    #     out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
    #     t1 = out
    #     out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
    #     t2 = out
    #     out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
    #     t3 = out

    #     ### Tokenized MLP Stage
    #     ### Stage 4

    #     out,H,W = self.patch_embed3(out)
    #     for i, blk in enumerate(self.block1):
    #         out = blk(out, H, W)
    #     out = self.norm3(out)
    #     out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    #     t4 = out

    #     ### Bottleneck

    #     out ,H,W= self.patch_embed4(out)
    #     for i, blk in enumerate(self.block2):
    #         out = blk(out, H, W)
    #     out = self.norm4(out)
    #     out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    #     ### Stage 4

    #     out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
    #     if t4.shape[2:] != out.shape[2:]:
    #        t4 = F.interpolate(t4, size=out.shape[2:], mode='bilinear', align_corners=True)

        
    #     out = torch.add(out,t4)
    #     _,_,H,W = out.shape
    #     out = out.flatten(2).transpose(1,2)
    #     for i, blk in enumerate(self.dblock1):
    #         out = blk(out, H, W)

    #     ### Stage 3
        
    #     out = self.dnorm3(out)
    #     out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    #     out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
    #     if t3.shape[2:] != out.shape[2:]:
    #        t3 = F.interpolate(t3, size=out.shape[2:], mode='bilinear', align_corners=True)

          
    #     out = torch.add(out,t3)
    #     _,_,H,W = out.shape
    #     out = out.flatten(2).transpose(1,2)
        
    #     for i, blk in enumerate(self.dblock2):
    #         out = blk(out, H, W)

    #     out = self.dnorm4(out)
    #     out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

    #     out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
    #     if t2.shape[2:] != out.shape[2:]:
    #        t2 = F.interpolate(t2, size=out.shape[2:], mode='bilinear', align_corners=True)
    #     out = torch.add(out,t2)


    #     out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
    #     if t1.shape[2:] != out.shape[2:]:
    #       t1 = F.interpolate(t1, size=out.shape[2:], mode='bilinear', align_corners=True)
    #     out = torch.add(out,t1)


    #     out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

    #     # return self.final(out)
    #     out = self.final(out)
    #     if out.shape[1] == 1:
    #         out = torch.sigmoid(out)  # For binary segmentation
    #     return out

    # def forward(self, x):
        
    #     out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
    #     t1 = out # b, c0, H/2, W/2

    #     out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
    #     t2 = out # b, c1, H/4, W/4 

    #     out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
    #     t3 = out # b, c2, H/8, W/8
        
    #     # out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        
    #     out,H,W = self.patch_embed3(out)
    #     for i, blk in enumerate(self.block1):
    #         out = blk(out, H, W)
    #     out = self.norm3(out)
    #     out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    #     t4 = out # b, c3, H/16, W/16
        
    #     out ,H,W= self.patch_embed4(out)
    #     for i, blk in enumerate(self.block2):
    #         out = blk(out, H, W)
    #     out = self.norm4(out)
    #     out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


    #     out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
    #     if t4.shape[2:] != out.shape[2:]:
    #        t4 = F.interpolate(t4, size=out.shape[2:], mode='bilinear', align_corners=True)

    #     if self.gt_ds: 
    #         gt_pre4 = self.gt_conv2(out4)
    #         t4 = self.GAB4(t5, t4, gt_pre4)
    #         gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
    #     else:t4 = self.GAB4(t5, t4)
    #     out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
    #     out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
    #     if self.gt_ds: 
    #         gt_pre3 = self.gt_conv3(out3)
    #         t3 = self.GAB3(t4, t3, gt_pre3)
    #         gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
    #     else: t3 = self.GAB3(t4, t3)
    #     out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
    #     out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
    #     if self.gt_ds: 
    #         gt_pre2 = self.gt_conv4(out2)
    #         t2 = self.GAB2(t3, t2, gt_pre2)
    #         gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
    #     else: t2 = self.GAB2(t3, t2)
    #     out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
    #     out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
    #     if self.gt_ds: 
    #         gt_pre1 = self.gt_conv5(out1)
    #         t1 = self.GAB1(t2, t1, gt_pre1)
    #         gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
    #     else: t1 = self.GAB1(t2, t1)
    #     out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
    #     out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
    #     if self.gt_ds:
    #         return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2), torch.sigmoid(gt_pre1)), torch.sigmoid(out0)
    #     else:
    #         return torch.sigmoid(out0)

    def forward(self, x):
        B = x.shape[0]

        # -------- Encoder: 3× CMRF stages --------
        # out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2)); t1 = out          # (B,16, H/2,  W/2)
        # out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2)); t2 = out        # (B,32, H/4,  W/4)
        # out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2)); t3 = out        # (B,128,H/8,  W/8)

        out = F.relu(self.pool1(self.encoder1(x)));        t1 = out   # (B,16, H/2,  W/2)
        out = F.relu(self.pool2(self.encoder2(out)));      t2 = out   # (B,32, H/4,  W/4)
        out = F.relu(self.pool3(self.encoder3(out)));      t3 = out   # (B,128,H/8,  W/8)

        # -------- Tok-MLP Stage 1 (H/8 -> H/16, 128->160) --------
        tok, H, W = self.patch_embed3(out)                    # (B, H/16*W/16, 160)
        for blk in self.block1:
            tok = blk(tok, H, W)
        tok = self.norm3(tok)
        t4 = tok.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()        # (B,160,H/16,W/16)

        # -------- Bottleneck Tok-MLP (H/16 -> H/32, 160->256) --------
        tok, H, W = self.patch_embed4(t4)                     # (B, H/32*W/32, 256)
        for blk in self.block2:
            tok = blk(tok, H, W)
        tok = self.norm4(tok)
        bot = tok.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()       # (B,256,H/32,W/32)

        # ================== Decoder with GAB ==================

        # ---- Level 4 decode (to 160, up to H/16), GAB4 with t4 ----
        out4 = F.relu(F.interpolate(self.dbn1(self.decoder1(bot)), scale_factor=(2, 2),
                                    mode='bilinear', align_corners=True))     # (B,160,H/16,W/16)
        if t4.shape[2:] != out4.shape[2:]:
            t4 = F.interpolate(t4, size=out4.shape[2:], mode='bilinear', align_corners=True)

        if self.use_gab:
            if self.gt_ds:
                gt4 = self.gt_conv4(out4)                                     # (B,1,H/16,W/16)
                t4 = self.GAB4(bot, t4, gt4)
                gt4_up = F.interpolate(gt4, scale_factor=16, mode='bilinear', align_corners=True)
            else:
                t4 = self.GAB4(bot, t4, None)
        out4 = out4 + t4                                                      # (B,160,H/16,W/16)
        xh3 = out4

        # ---- Level 3 token block + decode (to 128, up to H/8), GAB3 with t3 ----
        _, _, H, W = out4.shape
        tok = out4.flatten(2).transpose(1, 2)                                 # (B, H*W, 160)
        for blk in self.dblock1:
            tok = blk(tok, H, W)
        mid = self.dnorm3(tok).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,160,H/16,W/16)

        out3 = F.relu(F.interpolate(self.dbn2(self.decoder2(mid)), scale_factor=(2, 2),
                                    mode='bilinear', align_corners=True))     # (B,128,H/8,W/8)
        if t3.shape[2:] != out3.shape[2:]:
            t3 = F.interpolate(t3, size=out3.shape[2:], mode='bilinear', align_corners=True)

        if self.use_gab:
            if self.gt_ds:
                gt3 = self.gt_conv3(out3)                                     # (B,1,H/8,W/8)
                t3  = self.GAB3(xh3, t3, gt3)
                gt3_up = F.interpolate(gt3, scale_factor=8, mode='bilinear', align_corners=True)
            else:
                t3  = self.GAB3(xh3, t3, None)
        out3 = out3 + t3                                                      # (B,128,H/8,W/8)
        xh2 = out3

        # ---- Level 2 token block + decode (to 32, up to H/4), GAB2 with t2 ----
        _, _, H, W = out3.shape
        tok = out3.flatten(2).transpose(1, 2)                                 # (B, H*W, 128)
        for blk in self.dblock2:
            tok = blk(tok, H, W)
        mid = self.dnorm4(tok).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,128,H/8,W/8)

        out2 = F.relu(F.interpolate(self.dbn3(self.decoder3(mid)), scale_factor=(2, 2),
                                    mode='bilinear', align_corners=True))     # (B,32,H/4,W/4)
        if t2.shape[2:] != out2.shape[2:]:
            t2 = F.interpolate(t2, size=out2.shape[2:], mode='bilinear', align_corners=True)

        if self.use_gab:
            if self.gt_ds:
                gt2 = self.gt_conv2(out2)                                     # (B,1,H/4,W/4)
                t2  = self.GAB2(xh2, t2, gt2)
                gt2_up = F.interpolate(gt2, scale_factor=4, mode='bilinear', align_corners=True)
            else:
                t2  = self.GAB2(xh2, t2, None)
        out2 = out2 + t2                                                      # (B,32,H/4,W/4)
        xh1 = out2

        # ---- Level 1 decode (to 16, up to H/2), GAB1 with t1 ----
        out1 = F.relu(F.interpolate(self.dbn4(self.decoder4(out2)), scale_factor=(2, 2),
                                    mode='bilinear', align_corners=True))     # (B,16,H/2,W/2)
        if t1.shape[2:] != out1.shape[2:]:
            t1 = F.interpolate(t1, size=out1.shape[2:], mode='bilinear', align_corners=True)

        if self.use_gab:
            if self.gt_ds:
                gt1 = self.gt_conv1(out1)                                     # (B,1,H/2,W/2)
                t1  = self.GAB1(xh1, t1, gt1)
                gt1_up = F.interpolate(gt1, scale_factor=2, mode='bilinear', align_corners=True)
            else:
                t1  = self.GAB1(xh1, t1, None)
        out1 = out1 + t1                                                      # (B,16,H/2,W/2)

        # ---- Final up + head ----
        out0 = F.relu(F.interpolate(self.decoder5(out1), scale_factor=(2, 2),
                                    mode='bilinear', align_corners=True))     # (B,16,H,W)
        logits = self.final(out0)
        out = torch.sigmoid(logits) if logits.shape[1] == 1 else logits

        return out 
        # if self.use_gab and self.gt_ds:
        #     return (torch.sigmoid(gt4_up), torch.sigmoid(gt3_up),
        #             torch.sigmoid(gt2_up), torch.sigmoid(gt1_up)), out
        # return out

        # sdfsfsfs

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
    model = UNext_CMRF_GAB(num_classes=1, input_channels=3)
    model.eval()

    # Dummy input: B x C x H x W
    dummy_input = torch.randn(1, 3, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # print(f"✅ Forward pass successful! Output shape: {output.shape}")
    print(f"✅ Forward pass successful! Output shape: ")



#EOF