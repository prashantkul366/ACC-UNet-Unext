import torch.nn as nn
from vit_seg_modeling_KAN_fJNB import VisionTransformer, CONFIGS

class TransUNet_KAN_fJNB(nn.Module):
    def __init__(
        self,
        n_channels=1,
        n_classes=1,
        img_size=224,
        vit_name="R50-ViT-B_16",
    ):
        super().__init__()
        config_vit = CONFIGS[vit_name]

        config_vit.n_classes = n_classes
        config_vit.n_skip = 3
        config_vit.classifier = "seg"

        config_vit.decoder_channels = (256, 128, 64, 16)
        config_vit.skip_channels = [512, 256, 64, 16]

        config_vit.use_kan_ffn = True
        print("Using KAN MLP in TransUNet_KAN_fJNB")
        self.vit = VisionTransformer(
            config=config_vit,
            img_size=img_size,
            num_classes=config_vit.n_classes,
            zero_head=False,
            vis=False,
        )

    def forward(self, x):
        
        return self.vit(x)
