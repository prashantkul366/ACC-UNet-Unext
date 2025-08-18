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

from TinyU_Net import CMRF
# from nets.archs.TinyU_Net import CMRF



class NodeConv(nn.Module):
    """Conv block used at each UNet++ node after concatenation."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)  # compress concat width
        self.bn1  = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(out_ch)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.proj(x)))
        x = self.act2(self.bn2(self.conv(x)))
        return x


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

class UNext_CMRF_PP(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self, n_channels=3, n_classes=1,  deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        
        print("UNext CMRF Encoders Initiated")
        # self.encoder1 = nn.Conv2d(n_channels, 16, 3, stride=1, padding=1)  
        # self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        # self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.encoder1 = CMRF(n_channels, 16)
        self.encoder2 = CMRF(16, 32)
        self.encoder3 = CMRF(32, 128)

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
        ### Encoder
        ### Conv Stage

        ### Stage 1
        # out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        # t1 = out
        # ### Stage 2
        # out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        # t2 = out
        # ### Stage 3
        # out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        # t3 = out

        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        if t4.shape[2:] != out.shape[2:]:
           t4 = F.interpolate(t4, size=out.shape[2:], mode='bilinear', align_corners=True)

        
        out = torch.add(out,t4)
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

          
        out = torch.add(out,t3)
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        if t2.shape[2:] != out.shape[2:]:
           t2 = F.interpolate(t2, size=out.shape[2:], mode='bilinear', align_corners=True)
        out = torch.add(out,t2)


        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        if t1.shape[2:] != out.shape[2:]:
          t1 = F.interpolate(t1, size=out.shape[2:], mode='bilinear', align_corners=True)
        out = torch.add(out,t1)


        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        # return self.final(out)
        out = self.final(out)
        if out.shape[1] == 1:
            out = torch.sigmoid(out)  # For binary segmentation
        return out


class UNext_CMRF_PP_UNetPP(nn.Module):
    """
    UNet++-style nested decoder on top of your UNeXt+CMRF backbone.
    J = 2 (x_{i,1} and x_{i,2} for rows 0..2; x_{3,1} for row 3).
    """
    def __init__(self, n_channels=3, n_classes=1, img_size=224):
        super().__init__()
        self.n_classes = n_classes

        # ====== ENCODER (reuse your exact CMRF encoders) ======
        self.encoder1 = CMRF(n_channels, 16)   # row0 base
        self.encoder2 = CMRF(16, 32)           # row1 base
        self.encoder3 = CMRF(32, 128)          # row2 base

        # ====== TOKENIZED STAGES (reuse your exact UNeXt blocks) ======
        embed_dims = [128, 160, 256]
        norm_layer = nn.LayerNorm

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.block1 = nn.ModuleList([shiftedBlock(dim=embed_dims[1], num_heads=1, mlp_ratio=1)])
        self.block2 = nn.ModuleList([shiftedBlock(dim=embed_dims[2], num_heads=1, mlp_ratio=1)])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        # ====== UNet++ NODE CONVS (fixed concat widths for J=2) ======
        # row3 (14x14): x31 takes [x30(160), up(bottleneck 256)] -> 160
        self.node3_1 = NodeConv(in_ch=160 + 256, out_ch=160)

        # row2 (28x28): x21 takes [x20(128), up(x30 160)] -> 128
        self.node2_1 = NodeConv(in_ch=128 + 160, out_ch=128)
        #                x22 takes [x20(128), x21(128), up(x31 160)] -> 128
        self.node2_2 = NodeConv(in_ch=128 + 128 + 160, out_ch=128)

        # row1 (56x56): x11 takes [x10(32), up(x20 128)] -> 32
        self.node1_1 = NodeConv(in_ch=32 + 128, out_ch=32)
        #                x12 takes [x10(32), x11(32), up(x21 128)] -> 32
        self.node1_2 = NodeConv(in_ch=32 + 32 + 128, out_ch=32)

        # row0 (112x112): x01 takes [x00(16), up(x10 32)] -> 16
        self.node0_1 = NodeConv(in_ch=16 + 32, out_ch=16)
        #                 x02 takes [x00(16), x01(16), up(x11 32)] -> 16
        self.node0_2 = NodeConv(in_ch=16 + 16 + 32, out_ch=16)

        # ====== HEADS (deep supervision at row0) ======
        self.head_01 = nn.Conv2d(16, n_classes, kernel_size=1)
        self.head_02 = nn.Conv2d(16, n_classes, kernel_size=1)

    def _upsample_to(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=True)
        return x

    def _mlp_block(self, tokens, H, W, blocks, norm):
        for blk in blocks:
            tokens = blk(tokens, H, W)
        tokens = norm(tokens)
        tokens = tokens.reshape(tokens.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
        return tokens

    def forward(self, x):
        B, _, H0, W0 = x.shape  # expect 224x224

        # ====== ENCODER ROWS ======
        # row0 base x00 @112
        x00 = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))       # 16 @ 112x112
        # row1 base x10 @56
        x10 = F.relu(F.max_pool2d(self.encoder2(x00), 2, 2))     # 32 @ 56x56
        # row2 base x20 @28
        x20 = F.relu(F.max_pool2d(self.encoder3(x10), 2, 2))     # 128 @ 28x28

        # ====== TOKEN ROW (row3 base) + BOTTLENECK ======
        # from 28x28 -> patch_embed3 (stride 2) -> 14x14 tokens (embed 160)
        t, H3, W3 = self.patch_embed3(x20)                       # H3=W3=14
        x30 = self._mlp_block(t, H3, W3, self.block1, self.norm3)  # 160 @ 14x14

        # bottleneck: 14x14 -> patch_embed4 (stride 2) -> 7x7 tokens (embed 256)
        b, Hb, Wb = self.patch_embed4(x30)                       # Hb=Wb=7
        bottleneck = self._mlp_block(b, Hb, Wb, self.block2, self.norm4)  # 256 @ 7x7

        # ====== UNet++ NESTED NODES ======
        # row3: x31 = f([x30, up(bottleneck)])
        up_bottleneck_14 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)  # 7->14
        x31 = self.node3_1(torch.cat([x30, up_bottleneck_14], dim=1))    # 160 @ 14x14

        # row2:
        # x21 = f([x20, up(x30)])
        x30_up_28 = F.interpolate(x30, scale_factor=2, mode='bilinear', align_corners=True)               # 14->28
        x21 = self.node2_1(torch.cat([x20, x30_up_28], dim=1))             # 128 @ 28x28
        # x22 = f([x20, x21, up(x31)])
        x31_up_28 = F.interpolate(x31, scale_factor=2, mode='bilinear', align_corners=True)               # 14->28
        x22 = self.node2_2(torch.cat([x20, x21, x31_up_28], dim=1))        # 128 @ 28x28

        # row1:
        # x11 = f([x10, up(x20)])
        x20_up_56 = F.interpolate(x20, scale_factor=2, mode='bilinear', align_corners=True)               # 28->56
        x11 = self.node1_1(torch.cat([x10, x20_up_56], dim=1))             # 32  @ 56x56
        # x12 = f([x10, x11, up(x21)])
        x21_up_56 = F.interpolate(x21, scale_factor=2, mode='bilinear', align_corners=True)               # 28->56
        x12 = self.node1_2(torch.cat([x10, x11, x21_up_56], dim=1))        # 32  @ 56x56

        # row0:
        # x01 = f([x00, up(x10)])
        x10_up_112 = F.interpolate(x10, scale_factor=2, mode='bilinear', align_corners=True)              # 56->112
        x01 = self.node0_1(torch.cat([x00, x10_up_112], dim=1))            # 16  @ 112x112
        # x02 = f([x00, x01, up(x11)])
        x11_up_112 = F.interpolate(x11, scale_factor=2, mode='bilinear', align_corners=True)              # 56->112
        x02 = self.node0_2(torch.cat([x00, x01, x11_up_112], dim=1))       # 16  @ 112x112

        # ====== HEADS (deep supervision at row0) ======
        # Upsample to input size (224x224)
        logit_01 = self.head_01(x01)
        logit_02 = self.head_02(x02)

        logit_01 = F.interpolate(logit_01, size=(H0, W0), mode='bilinear', align_corners=True)
        logit_02 = F.interpolate(logit_02, size=(H0, W0), mode='bilinear', align_corners=True)

        if self.n_classes == 1:
            logit_01 = torch.sigmoid(logit_01)
            logit_02 = torch.sigmoid(logit_02)

        # Return both (train) or their mean (simple inference). Choose what you prefer:
        return {"out_mean": (logit_01 + logit_02) / 2, "out_01": logit_01, "out_02": logit_02}




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

# if __name__ == '__main__':
#     # Sanity check
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = UNext_CMRF(num_classes=1, input_channels=3)
#     model.eval()

#     # Dummy input: B x C x H x W
#     dummy_input = torch.randn(1, 3, 224, 224)

#     # Forward pass
#     with torch.no_grad():
#         output = model(dummy_input)

#     print(f"✅ Forward pass successful! Output shape: {output.shape}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNext_CMRF_PP_UNetPP(n_channels=3, n_classes=1).to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy_input)

    print("✅ UNet++ nested decoder ok")
    for k, v in out.items():
        print(k, v.shape)


#EOF