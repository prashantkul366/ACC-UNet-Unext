import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ===============================
# DDCM BLOCK
# ===============================
class DDCMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, rates):
        super().__init__()

        self.rates = rates

        self.convs = nn.ModuleList()
        cur_ch = in_ch

        for r in rates:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(cur_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                    nn.PReLU(),
                    nn.BatchNorm2d(out_ch)
                )
            )
            cur_ch += out_ch     # dense concat

        self.merge = nn.Sequential(
            nn.Conv2d(cur_ch, out_ch, 1, bias=False),
            nn.PReLU(),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):

        features = x
        for conv in self.convs:
            out = conv(features)
            features = torch.cat([features, out], dim=1)

        out = self.merge(features)
        return out


# ===============================
# Light DDCM NET
# ===============================
class LightDDCMNet(nn.Module):

    def __init__(self, in_channels=1):
        super().__init__()

        # ---- input expansion conv ----
        self.input_conv = nn.Conv2d(in_channels, 3 - in_channels, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(3)
        self.prelu0 = nn.PReLU()

        # ---- ResNet backbone ----
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3      # stop before layer4 (paper)
        )

        enc_out_ch = 1024

        # ---- DDCM modules ----
        self.ddcm1 = DDCMBlock(enc_out_ch, 256, rates=[1,2,3,4])
        self.ddcm2 = DDCMBlock(256, 128, rates=[1])

        # ---- final head ----
        self.final_conv = nn.Conv2d(128, 1, 1)

    def forward(self, x):

        # create pseudo 3 channel
        if x.shape[1] == 1:
            x2 = self.input_conv(x)
            x = torch.cat([x, x2], dim=1)

        x = self.prelu0(self.bn0(x))

        # encoder
        x = self.encoder(x)

        # decoder
        x = self.ddcm1(x)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

        x = self.ddcm2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        x = self.final_conv(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return torch.sigmoid(x)