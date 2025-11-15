import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

from .vit_seg_modeling_KAN_fJNB import Transformer, CONFIGS


class SegViT_fKAN(nn.Module):
    """
    ViT-fKAN encoder + ResNet hybrid backbone, with a SegMamba-style UNETR decoder.

    - Encoder:
        * R50-ViT-B_16 (hybrid ResNet+ViT) with KAN MLP blocks.
        * Uses Transformer(...) from vit_seg_modeling_KAN_fJNB.
        * We use the ViT tokens as bottleneck, and ResNet features as skips.

    - Decoder:
        * 2D UNETR-style (UnetrBasicBlock + UnetrUpBlock), similar U-shape
          as SegMamba but in 2D (since BUSI is 2D).
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        img_size=224,
        vit_name="R50-ViT-B_16",
        feat_size=(64, 128, 256, 512),
        norm_name="instance",
        res_block=True,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.img_size = img_size
        self.feat_size = list(feat_size)
        print(f"SegViT_fKAN using")

        # ---- ViT-fKAN encoder config ----
        config_vit = CONFIGS[vit_name]
        config_vit.classifier = "seg"
        # enable KAN FFN inside blocks (already used in Block)
        config_vit.use_kan_ffn = True

        self.hidden_size = config_vit.hidden_size  # e.g. 768

        # Transformer = patch embedding (ResNet+ViT) + encoder (with KANMLP)
        self.transformer = Transformer(config_vit, img_size=img_size, vis=False)

        # We'll use ResNet feature maps as skips.
        # In original TransUNet, these are given by config.skip_channels.
        # They are the channels of hybrid ResNet features list.
        res_skip_channels = config_vit.skip_channels  # e.g. [256, 512, 1024, 2048]

        # Project ResNet features to our desired feat_size for the UNETR decoder
        # self.res_proj = nn.ModuleList([
        #     nn.Conv2d(res_skip_channels[i], self.feat_size[i], kernel_size=1)
        #     for i in range(4)
        # ])
        self.res_proj = nn.ModuleList([
            nn.Conv2d(res_skip_channels[i], self.feat_size[i], kernel_size=1)
            for i in range(3)   # we only use f1,f2,f3
        ])


        # Project token embedding (ViT output) into a spatial "bottleneck" feat map
        # and then into feat_size[-1] channels.
        self.tokens_to_map = nn.Conv2d(self.hidden_size, self.feat_size[3], kernel_size=3, padding=1)

        # ---- UNETR style encoder blocks (just refinement on top of ResNet feats) ----
        # enc1 will operate on input image
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # these will refine the projected ResNet skips
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # bottleneck refinement of the ViT feature map
        self.encoder5 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # ---- Decoder (SegMamba-style but 2D) ----
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=2,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.out = UnetOutBlock(
            spatial_dims=2,
            in_channels=self.feat_size[0],
            out_channels=self.out_chans,
        )

        print("SegViT_fKAN (ResNet-ViT-KAN encoder + SegMamba UNETR decoder) initialized.")

    def forward(self, x_in):
        """
        x_in: [B, C, H, W] (BUSI ultrasound)

        - Ensure 3 channels for ResNet-ViT if needed.
        - Get ViT tokens + ResNet features via Transformer.
        - Project tokens into a spatial bottleneck.
        - Project ResNet skips and feed them through UNETR encoder+decoder.
        """
        B, C, H, W = x_in.shape

        # ---- handle grayscale input for R50 backbone ----
        if x_in.size(1) == 1:
            x_rgb = x_in.repeat(1, 3, 1, 1)  # [B,3,H,W]
        else:
            x_rgb = x_in

        # ---- ViT-fKAN encoder ----
        # transformer returns: encoded_tokens [B, N, hidden], attn_weights, resnet_features (list of 4)
        encoded_tokens, attn_weights, res_features = self.transformer(x_rgb)

        # tokens -> spatial map (like TransUNet's DecoderCup first step)
        B_, N, D = encoded_tokens.shape
        h = w = int(math.sqrt(N))
        x_tokens = encoded_tokens.permute(0, 2, 1).contiguous().view(B_, D, h, w)
        x_bottleneck = self.tokens_to_map(x_tokens)  # [B, feat_size[3], h, w]

        # ---- ResNet skips projected to feat_size[i] ----
        # res_features is a list of 4 feature maps: coarse -> fine
        # we want them from shallow to deep to match feat_size:
        # res_features[0] -> highest resolution
        # depending on implementation, you might need to reverse; adjust if needed.
        f1 = self.res_proj[0](res_features[0])  # [B, feat_size[0], H/4, W/4]  (depending on ResNet)
        f2 = self.res_proj[1](res_features[1])  # [B, feat_size[1], ...]
        f3 = self.res_proj[2](res_features[2])  # [B, feat_size[2], ...]
        

        #You may need to ensure these resolutions line up with your up/down path;
        #if not, you can F.interpolate(f_i, size=...) to correct.

        # ---- UNETR-style encoder path ----
        enc1 = self.encoder1(x_in)    # full-res input
        enc2 = self.encoder2(f1)
        enc3 = self.encoder3(f2)
        enc4 = self.encoder4(f3)
        enc_hidden = self.encoder5(x_bottleneck)  # bottleneck

        print("enc1:", enc1.shape)
        print("enc2:", enc2.shape)
        print("enc3:", enc3.shape)
        print("enc4:", enc4.shape)
        print("enc_hidden:", enc_hidden.shape)

        # ---- Decoder path (SegMamba-like) ----
        dec3 = self.decoder5(enc_hidden, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)

        logits = self.out(out)
        return logits
