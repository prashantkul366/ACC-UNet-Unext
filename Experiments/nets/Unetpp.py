import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x2"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    """UNet++ Model based on MrGiovanni's GitHub implementation"""
    def __init__(self, n_channels=3, n_classes=1, deep_supervision=False):
        super(UNetPlusPlus, self).__init__()

        self.deep_supervision = deep_supervision

        self.conv0_0 = ConvBlock(n_channels, 64)
        self.conv1_0 = ConvBlock(64, 128)
        self.conv2_0 = ConvBlock(128, 256)
        self.conv3_0 = ConvBlock(256, 512)
        self.conv4_0 = ConvBlock(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.up1_0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up2_0 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3_0 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up4_0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.conv0_1 = ConvBlock(64 + 64, 64)
        self.conv1_1 = ConvBlock(128 + 128, 128)
        self.conv2_1 = ConvBlock(256 + 256, 256)
        self.conv3_1 = ConvBlock(512 + 512, 512)

        self.conv0_2 = ConvBlock(64*2 + 64, 64)
        self.conv1_2 = ConvBlock(128*2 + 128, 128)
        self.conv2_2 = ConvBlock(256*2 + 256, 256)

        self.conv0_3 = ConvBlock(64*3 + 64, 64)
        self.conv1_3 = ConvBlock(128*3 + 128, 128)

        self.conv0_4 = ConvBlock(64*4 + 64, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        # Add activation based on number of classes
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()  # Needed for Binary Classification (BCE Loss)
        else:
            self.last_activation = None  # Multi-class tasks typically use raw logits (for CrossEntropy)


    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_0(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_0(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_0(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_0(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_0(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_0(x1_3)], 1))

        logits = self.final_conv(x0_4)  # Compute logits

        # Apply activation if required
        if self.last_activation is not None:
            logits = self.last_activation(logits)  # Sigmoid for BCE Loss

        return logits

    
