# import torch
# from thop import profile
# from nets.segmamba_hybrid_gsc_KAN_PE_ds_CrossAttn_HSLCA_SpatialMamba import SegMamba as Segmamba_hybrid_gsc_KAN_PE_ds_CrossAttn_HSLCA_SpatialMamba

# # -----------------------------
# # Create model
# # -----------------------------
# model = Segmamba_hybrid_gsc_KAN_PE_ds_CrossAttn_HSLCA_SpatialMamba(
#             in_chans=3, out_chans=1, depths=[2, 2, 2, 2],
#             feat_size=[48, 96, 192, 384], spatial_dims=3,).cuda()

# model.eval()

# # -----------------------------
# # Dummy inputs
# # -----------------------------
# B = 1
# H = 256
# W = 256

# # dummy_image = torch.randn(B, 1, H, W).cuda()
# dummy_image = torch.randn(B, 3, H, W).cuda()

# dummy_text = [
#     "lung infection with ground glass opacity"
# ] * B

# # -----------------------------
# # THOP profiling
# # -----------------------------
# # macs, params = profile(
# #     model,
# #     inputs=(dummy_image, dummy_text),
# #     verbose=False
# # )

# # def count_trainable_params(model):
# #     return sum(
# #         p.numel()
# #         for n, p in model.named_parameters()
# #         if p.requires_grad and "text_encoder" not in n
# #     )

# # params = count_trainable_params(model)

# # print(f"Vision Params (no text encoder): {params/1e6:.2f} M")

# # print(f"Encoder params{sum(p.numel() for p in model.text_encoder.parameters()) / 1e6}")
# # print(f"Params: {params/1e6:.2f} M")
# # print(f"MACs: {macs/1e9:.2f} G")
# # print(f"FLOPs: {(macs*2)/1e9:.2f} G")


# # REMOVE TEXT ENCODER FROM FLOPs
# model.text_encoder = torch.nn.Identity()

# dummy_img = torch.randn(1, 1, 256, 256).cuda()
# dummy_text = None

# def forward_for_thop(x):
#     out = model(x, dummy_text)
#     if isinstance(out, tuple):
#         return out[0]
#     return out

# macs, params = profile(
#     forward_for_thop,
#     inputs=(dummy_img,),
#     verbose=False
# )

# print("====== FINAL REPORT ======")
# print(f"Params (Vision only): {params/1e6:.2f} M")
# print(f"GFLOPs: {(macs*2)/1e9:.2f} G")


import torch
import torch.nn as nn
from thop import profile

from nets.segmamba_hybrid_gsc_KAN_PE_ds_CrossAttn_HSLCA_SpatialMamba import (
    SegMamba as SegMambaModel
)

# =====================================================
# Fake Text Encoder (VERY IMPORTANT)
# =====================================================
class FakeTextEncoder(nn.Module):
    """
    Replaces ClinicalBERT during FLOPs computation.
    Produces dummy token embeddings.
    """
    def forward(self, texts):
        B = 1 if texts is None else len(texts)

        # ClinicalBERT output shape
        # (B, Tokens, 768)
        return torch.randn(
            B,
            16,        # token length (approx)
            768,
            device="cuda"
        )


# =====================================================
# Create Model
# =====================================================
model = SegMambaModel(
    in_chans=1,
    out_chans=1,
    depths=[2,2,2,2],
    feat_size=[48,96,192,384],
    spatial_dims=3,
).cuda()

model.eval()

# Replace heavy ClinicalBERT
print("\nReplacing ClinicalBERT with Fake Encoder...")
model.text_encoder = FakeTextEncoder()


# =====================================================
# THOP Wrapper
# =====================================================
class THOPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        dummy_text = ["dummy report"]
        out = self.model(x, dummy_text)

        if isinstance(out, tuple):
            return out[0]

        return out


wrapped_model = THOPWrapper(model).cuda()
wrapped_model.eval()


# =====================================================
# Dummy Image
# =====================================================
dummy_image = torch.randn(1, 1, 256, 256).cuda()


# =====================================================
# Profile
# =====================================================
print("\nProfiling model...")

macs, params = profile(
    wrapped_model,
    inputs=(dummy_image,),
    verbose=False
)

flops = macs * 2


# =====================================================
# Results
# =====================================================
print("\n============== MODEL COMPLEXITY ==============")
print("Input Size : 1×1×256×256")
print(f"Parameters : {params/1e6:.2f} M")
print(f"MACs       : {macs/1e9:.2f} G")
print(f"FLOPs      : {flops/1e9:.2f} G")
print("==============================================")