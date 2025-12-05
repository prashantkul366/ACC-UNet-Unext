import torch
import torch.nn.functional as F
from torch import nn
import cv2
import numpy as np
import os

# -------------------------------
# Grad-CAM for segmentation
# -------------------------------
class SegGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # forward hook
        self.target_layer.register_forward_hook(
            lambda module, inp, out: self.save_activation(out)
        )

        # backward hook
        self.target_layer.register_full_backward_hook(
            lambda module, grad_in, grad_out: self.save_gradient(grad_out[0])
        )

    def save_activation(self, activation):
        self.activations = activation

    def save_gradient(self, gradient):
        self.gradients = gradient

    def generate_cam(self, input_tensor):
        self.model.zero_grad()

        out = self.model(input_tensor)
        score = out.sum()         # backprop wrt all pixels
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam[0, 0].cpu().numpy()


# -------------------------------
# Heatmap overlay
# -------------------------------
def overlay_heatmap(cam, img):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.5 * heatmap + 0.5 * img).astype(np.uint8)
    return overlay


# -------------------------------
# Load and resize image
# -------------------------------
def load_image_for_model(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, (256, 256))
    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    tensor = tensor.unsqueeze(0)
    return img, tensor


# -------------------------------
# PROCESS FOLDER (MAIN)
# -------------------------------
def run_compare():
    # ---------------- Hardcoded paths ----------------
    model1_path = '/content/drive/MyDrive/Prashant/ACC-UNet-Unext/BUSI_80-20/UNext_CMRF_GS_Wavelet/session1/model/best_model-UNext_CMRF_GS_Wavelet.pth.tar'
    model2_path = '/content/drive/MyDrive/Prashant/ACC-UNet-Unext/BUSI_80-20/UNext_CMRF_GS_Wavelet/session1/model/best_model-UNext_CMRF_GS_Wavelet_rKAN.pth.tar'
    input_folder = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/test/images'
    output_folder = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/gradcam'
    # -------------------------------------------------

    os.makedirs(output_folder, exist_ok=True)

    # Load models
    model1 = torch.load(model1_path, map_location="cpu"); model1.eval()
    model2 = torch.load(model2_path, map_location="cpu"); model2.eval()

    # Target layer (you can change)
    target_layer_1 = list(model1.modules())[-2]
    target_layer_2 = list(model2.modules())[-2]

    cam1_gen = SegGradCAM(model1, target_layer_1)
    cam2_gen = SegGradCAM(model2, target_layer_2)

    device1 = next(model1.parameters()).device
    device2 = next(model2.parameters()).device

    # Process every image
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_folder, filename)
        print(f"Processing → {img_path}")
        orig_img, tensor = load_image_for_model(img_path)

        tensor1 = tensor.to(device1)
        tensor2 = tensor.to(device2)

        # CAM for both models
        cam1 = cam1_gen.generate_cam(tensor1)
        cam2 = cam2_gen.generate_cam(tensor2)

        # Resize CAM
        H, W = orig_img.shape[:2]
        cam1_resized = cv2.resize(cam1, (W, H))
        cam2_resized = cv2.resize(cam2, (W, H))

        # Overlay
        overlay1 = overlay_heatmap(cam1_resized, orig_img)
        overlay2 = overlay_heatmap(cam2_resized, orig_img)

        # Side-by-side comparison
        comparison = np.concatenate([overlay1, overlay2], axis=1)

        out_path = os.path.join(output_folder, f"{filename}_compare.png")
        print(f"Saving → {out_path}")
        cv2.imwrite(out_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        print(f"Saved → {out_path}")


if __name__ == "__main__":
    run_compare()
