import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os

# ---------------------------------------------------
# 1) IMPORT YOUR MODEL CLASSES HERE
# ---------------------------------------------------
from nets.archs.UNext_CMRF_GS_wavelet import UNext_CMRF_GS_Wavelet
from nets.archs.UNext_CMRF_GS_wavelet_rkan import UNext_CMRF_GS_Wavelet_rKAN
# ---------------------------------------------------


# -------------------------------
# Grad-CAM for segmentation
# -------------------------------
class SegGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(
            lambda module, inp, out: self.save_activation(out)
        )

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
        score = out.sum()
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam[0, 0].cpu().numpy()


def overlay_heatmap(cam, img):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (0.5 * heatmap + 0.5 * img).astype(np.uint8)


def load_image_for_model(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, (256, 256))
    t = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    t = t.unsqueeze(0)
    return img, t


def load_checkpoint_model(model_path, model_class):
    print(f"Loading checkpoint → {model_path}")
    ckpt = torch.load(model_path, map_location="cpu")

    model = model_class()
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


# -------------------------------
# MAIN COMPARISON
# -------------------------------
def run_compare():

    # model1_path = '/content/drive/MyDrive/.../best_model-UNext_CMRF_GS_Wavelet.pth.tar'
    # model2_path = '/content/drive/MyDrive/.../best_model-UNext_CMRF_GS_Wavelet_rKAN.pth.tar'

    # input_folder = '/content/drive/MyDrive/.../test/images'
    # output_folder = '/content/drive/MyDrive/.../gradcam_compare'

    # ---------------- Hardcoded paths ----------------
    model1_path = '/content/drive/MyDrive/Prashant/ACC-UNet-Unext/BUSI_80-20/UNext_CMRF_GS_Wavelet/session1/model/best_model-UNext_CMRF_GS_Wavelet.pth.tar'
    model2_path = '/content/drive/MyDrive/Prashant/ACC-UNet-Unext/BUSI_80-20/UNext_CMRF_GS_Wavelet/session1/model/best_model-UNext_CMRF_GS_Wavelet_rKAN.pth.tar'
    input_folder = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/test/images'
    output_folder = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/gradcam'
    # -------------------------------------------------
   

    os.makedirs(output_folder, exist_ok=True)

    # ------------ Load models properly -----------------
    model1 = load_checkpoint_model(model1_path, UNext_CMRF_GS_Wavelet)
    model2 = load_checkpoint_model(model2_path, UNext_CMRF_GS_Wavelet_rKAN)

    # ------------ choose target layer -------------------
    target_layer_1 = list(model1.modules())[-2]
    target_layer_2 = list(model2.modules())[-2]

    cam1 = SegGradCAM(model1, target_layer_1)
    cam2 = SegGradCAM(model2, target_layer_2)

    for filename in os.listdir(input_folder):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(input_folder, filename)
        print(f"\nProcessing → {img_path}")

        orig_img, tensor = load_image_for_model(img_path)

        # CAM generation
        cam_img1 = cam1.generate_cam(tensor)
        cam_img2 = cam2.generate_cam(tensor)

        H, W = orig_img.shape[:2]

        cam1_res = cv2.resize(cam_img1, (W, H))
        cam2_res = cv2.resize(cam_img2, (W, H))

        overlay1 = overlay_heatmap(cam1_res, orig_img)
        overlay2 = overlay_heatmap(cam2_res, orig_img)

        combined = np.concatenate([overlay1, overlay2], axis=1)

        save_path = os.path.join(output_folder, f"{filename}_compare.png")
        cv2.imwrite(save_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        print(f"Saved → {save_path}")


if __name__ == "__main__":
    run_compare()
