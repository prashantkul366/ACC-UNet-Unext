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
# 1. Create Model
# =====================================================
model = SegMambaModel(
    in_chans=1,                 # change if RGB
    out_chans=1,
    depths=[2, 2, 2, 2],
    feat_size=[48, 96, 192, 384],
    spatial_dims=3,
).cuda()

model.eval()


# =====================================================
# 2. REMOVE TEXT ENCODER FROM PROFILING
# (Standard practice in VLM papers)
# =====================================================
print("\nRemoving ClinicalBERT from FLOPs calculation...")
model.text_encoder = nn.Identity()


# =====================================================
# 3. THOP Wrapper
# (handles tuple outputs + text input)
# =====================================================
class THOPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # text=None because encoder removed
        out = self.model(x, None)

        # deep supervision returns tuple
        if isinstance(out, tuple):
            return out[0]

        return out


wrapped_model = THOPWrapper(model).cuda()
wrapped_model.eval()


# =====================================================
# 4. Dummy Input
# =====================================================
B = 1
H = 256
W = 256

dummy_image = torch.randn(B, 3, H, W).cuda()


# =====================================================
# 5. FLOPs + Params
# =====================================================
print("\nProfiling model...")

macs, params = profile(
    wrapped_model,
    inputs=(dummy_image,),
    verbose=False
)

flops = macs * 2


# =====================================================
# 6. Print Results
# =====================================================
print("\n================ MODEL COMPLEXITY ================")
print(f"Input Size : 1 x 1 x {H} x {W}")
print(f"Parameters : {params/1e6:.2f} M")
print(f"MACs       : {macs/1e9:.2f} G")
print(f"FLOPs      : {flops/1e9:.2f} G")
print("==================================================")