import torch
from thop import profile
from nets.segmamba_hybrid_gsc_KAN_PE_ds_CrossAttn_HSLCA_SpatialMamba import SegMamba as Segmamba_hybrid_gsc_KAN_PE_ds_CrossAttn_HSLCA_SpatialMamba

# -----------------------------
# Create model
# -----------------------------
model = Segmamba_hybrid_gsc_KAN_PE_ds_CrossAttn_HSLCA_SpatialMamba(
            in_chans=3, out_chans=1, depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384], spatial_dims=3,).cuda()

model.eval()

# -----------------------------
# Dummy inputs
# -----------------------------
B = 1
H = 256
W = 256

# dummy_image = torch.randn(B, 1, H, W).cuda()
dummy_image = torch.randn(B, 3, H, W).cuda()

dummy_text = [
    "lung infection with ground glass opacity"
] * B

# -----------------------------
# THOP profiling
# -----------------------------
macs, params = profile(
    model,
    inputs=(dummy_image, dummy_text),
    verbose=False
)

print(f"Encoder params{sum(p.numel() for p in model.text_encoder.parameters()) / 1e6}")
print(f"Params: {params/1e6:.2f} M")
print(f"MACs: {macs/1e9:.2f} G")
print(f"FLOPs: {(macs*2)/1e9:.2f} G")