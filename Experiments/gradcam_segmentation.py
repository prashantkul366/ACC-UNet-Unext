"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
import pickle
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os
# from utils import AverageMeter
# from thop import profile
import pandas as pd
import time

try:
    import distutils.version 
except Exception:
    import sys, types
    import packaging.version as pv
    dv = types.ModuleType("distutils.version")
    class LooseVersion(pv.Version): 
        pass
    sys.modules['distutils'] = types.ModuleType("distutils")
    sys.modules['distutils.version'] = dv
    dv.LooseVersion = LooseVersion

from thop import profile


# from nets.ACC_UNet import ACC_UNet
from nets.UCTransNet import UCTransNet
from nets.UNet_base import UNet_base
from nets.SMESwinUnet import SMESwinUnet
from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet
from nets.ACC_UNet import ACC_UNet
import json
from utils import *
import cv2


##################### NEW ARCHS ######################

from nets.UNext import UNext
# from nets.archs.archs_InceptionNext_MLFC import UNext_InceptionNext_MLFC
from nets.archs.UNext_CMRF import UNext_CMRF
from nets.archs.UNext_CMRF_enc_dec import UNext_CMRF_enc_dec
from nets.archs.UNext_CMRF_enc_MLFC import UNext_CMRF_enc_MLFC
from nets.archs.UNext_CMRF_enc_dec_MLFC import UNext_CMRF_enc_dec_MLFC
from nets.archs.UNext_CMRF_enc_CSSE import UNext_CMRF_enc_CSSE
# from nets.archs.UNext_CMRF_PP import UNext_CMRF_PP_UNetPP
from nets.archs.UNext_CMRF_dense_skip import UNext_CMRF_Dense_Skip

from nets.archs.UNext_CMRF_GAB import UNext_CMRF_GAB
from nets.archs.UNext_CMRF_GS import UNext_CMRF_GS

from nets.TransUNet import TransUNet
from nets.archs.u_kan import UKAN
from nets.archs.UNext_CMRF_GS_wavelet import UNext_CMRF_GS_Wavelet
from nets.archs.UNext_CMRF_GS_wavelet_OD import UNext_CMRF_GS_Wavelet_OD

from nets.archs.UNext_CMRF_GAB_wavelet import UNext_CMRF_GAB_Wavelet   
from nets.archs.UNext_CMRF_GAB_wavelet_OD import UNext_CMRF_GAB_Wavelet_OD 
from nets.archs.UNext_CMRF_GS_wavelet_hd import UNext_CMRF_GS_Wavelet_hd

from nets.archs.UNext_CMRF_BS_GS_wavelet import UNext_CMRF_BS_GS_Wavelet
from nets.archs.UNext_CMRF_BSRB_GS_wavelet import UNext_CMRF_BSRB_GS_Wavelet

from nets.archs.UNext_CMRF_BSRB_GS import UNext_CMRF_BSRB_GS

from nets.archs.UNext_CMRF_GS_wavelet_rkan import UNext_CMRF_GS_Wavelet_rKAN

# from nets.segmamba_hybrid_gsc_rm_fkan import SegMamba as Segmamba_hybrid_gsc_rm_fkan
# from nets.segmamba import SegMamba
from nets.segmamba_hybrid_gsc_KAN_PE_ds import SegMamba as Segmamba_hybrid_gsc_KAN_PE_ds
######################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def show_image_with_dice(predict_save, labs, save_path):

    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))


    return dice_pred, iou_pred


def get_final_prob(model_out, n_classes=1):
    """
    Return ONLY the final prediction as probabilities [B,C,H,W].
    Works whether the model returns a tensor or any nested tuple/list.
    """
    # unwrap nested structures and take the LAST tensor as the final head
    def _collect(x):
        if torch.is_tensor(x):
            return [x]
        if isinstance(x, (list, tuple)):
            out = []
            for it in x:
                out += _collect(it)
            return out
        return []

    tensors = _collect(model_out)
    if not tensors:
        raise TypeError(f"No tensors in model_out (type={type(model_out)})")
    final = tensors[-1]                    # <- final head logits/probs

    # ensure [B,C,H,W]
    if final.ndim == 3:                    # [B,H,W] -> [B,1,H,W]
        final = final.unsqueeze(1)

    # probabilities
    if n_classes == 1:
        final = torch.sigmoid(final)       # safe even if already sigmoided
    else:
        final = F.softmax(final, dim=1)
    return final


import torch.nn.functional as F

class SegGradCAM:
    """
    Grad-CAM for (2D or 3D) segmentation models.

    - Works with 3D features [B,C,D,H,W] (we collapse D; for you D=1).
    - Works with nested outputs (tuple/list) and deep supervision. We use the *main* output.
    - Treats final CAM as 2D [H,W] for overlay.

    Usage:
        use_model = model.module if isinstance(model, nn.DataParallel) else model
        target_layer = use_model.decoder1   # for SegMamba
        cam = SegGradCAM(use_model, target_layer)
        cam_map = cam(input_tensor)  # numpy (H,W) in [0,1]
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out: feature maps (4D or 5D)
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0]: same shape as activations
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def _clear(self):
        self.gradients = None
        self.activations = None

    def _extract_main_logits(self, model_output):
        """
        Handle different output types:
        - tensor: use directly
        - tuple/list: assume first is main output (for SegMamba: out_main, ds1, ds2, ds3)
        """
        if isinstance(model_output, (list, tuple)):
            out_main = model_output[0]
        else:
            out_main = model_output

        # out_main can be [B,C,H,W] or [B,C,D,H,W]
        if out_main.ndim == 5 and out_main.shape[2] == 1:
            # [B,C,1,H,W] -> [B,C,H,W]
            out_main = out_main.squeeze(2)
        return out_main  # [B,C,H,W]

    def __call__(self, input_tensor, target_score=None, retain_graph=False):
        """
        input_tensor: [B,C,H,W] on CUDA
        returns: CAM as numpy array [H,W] normalized to [0,1]
        """
        self._clear()
        self.model.zero_grad()

        # Forward pass
        out = self.model(input_tensor)
        out_main = self._extract_main_logits(out)  # [B,C,H,W]

        # Choose scalar score to backprop
        if target_score is None:
            # For binary segmentation C=1: use mean of logits
            # For multi-class: you can adapt to specific channel
            if out_main.shape[1] == 1:
                score = out_main[:, 0, :, :].mean()
            else:
                # simple choice: mean over argmax channel
                ch = out_main.argmax(dim=1)  # [B,H,W]
                gathered = out_main.gather(1, ch.unsqueeze(1)).squeeze(1)  # [B,H,W]
                score = gathered.mean()
        else:
            score = target_score

        score.backward(retain_graph=retain_graph)

        # Activations & gradients from target layer
        acts = self.activations      # [B,C,H,W] or [B,C,D,H,W]
        grads = self.gradients

        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients. "
                               "Check that target_layer is used in the forward pass.")

        # Support 3D (D,H,W) or 2D (H,W) features
        if acts.ndim == 5:
            # [B,C,D,H,W] -> global avg over D,H,W
            spatial_dims = (2, 3, 4)
        elif acts.ndim == 4:
            # [B,C,H,W] -> global avg over H,W
            spatial_dims = (2, 3)
        else:
            raise RuntimeError(f"Unsupported activation dim: {acts.shape}")

        # Weights: global average pooled gradients
        weights = grads.mean(dim=spatial_dims, keepdim=True)  # [B,C,1,1] or [B,C,1,1,1]

        # Weighted sum over channels
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,...]

        # ReLU
        cam = F.relu(cam)

        # If 5D, collapse depth -> [B,1,H,W]
        if cam.ndim == 5:
            cam = cam.mean(dim=2)  # average over D

        # Resize CAM to match input spatial size
        cam = F.interpolate(
            cam,
            size=out_main.shape[-2:],  # (H,W)
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()  # [H,W]

        # Normalize to [0,1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam


# def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
# def vis_and_save_heatmap(model, input_img, img_RGB, labs,
#                          vis_save_path, dice_pred, dice_ens,
#                          pred_img_path):
# def vis_and_save_heatmap(model, input_img, img_RGB, labs,vis_save_path, dice_pred, dice_ens,
#                          mask_dir, side_dir):
def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens,
                         mask_dir, side_dir, cam_dir, gradcam=None):


    model.eval()

    start_time = time.time()
    output = model(input_img.cuda())

    if isinstance(output, (list, tuple)):
        output_main = output[0]
    else:
        output_main = output
    
    if output_main.ndim == 5 and output_main.shape[2] == 1:
        output_main = output_main.squeeze(2)   # [B,C,H,W]

    # Convert logits -> probabilities for binary
    if output_main.shape[1] == 1:
        prob = torch.sigmoid(output_main)
    else:
        prob = F.softmax(output_main, dim=1)


    #  keep final head only
    # if isinstance(output, (tuple, list)):
    #     # ((aux... ), out)  -> take out
    #     # (aux1, aux2, ..., out) -> take last
    #     output = output[1] if (len(output) == 2 and isinstance(output[0], (tuple, list))) else output[-1]

    # # ensure [B,C,H,W]
    # if output.ndim == 3:
    #     output = output.unsqueeze(1)

    # convert to probabilities
    # output = torch.sigmoid(output) if output.shape[1] == 1 else F.softmax(output, dim=1)

    end_time = time.time()
    gpu_time_meter.update(end_time - start_time, input_img.size(0))

    # pred_class = (output > 0.5).float()
    # predict_save = pred_class[0, 0].cpu().numpy()   # squeeze channel; already HxW
    # (Delete the reshape line below; no longer needed)

    pred_class = (prob > 0.5).float()
    predict_save = pred_class[0, 0].cpu().numpy()   # [H,W]

    # pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    # predict_save = pred_class[0].cpu().data.numpy()

    # predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    
    input_img_np = input_img[0].transpose(0, -1).cpu().detach().numpy()  # [H,W,C]
    output_map = prob[0, 0].cpu().detach().numpy()                       # [H,W]

    # convert labs to 2D numpy ground truth
    if torch.is_tensor(labs):
        labs_np = labs.cpu().numpy()
    else:
        labs_np = np.array(labs)

    gt = np.squeeze(labs_np)  # -> [H,W]

    dice_pred_tmp, iou_tmp = show_image_with_dice(
        predict_save,
        gt,
        save_path=vis_save_path + '_predict' + model_type + '.jpg'
    )
    # input_img_np = input_img[0].transpose(0, -1).cpu().detach().numpy()  # [H,W,C]
    # output_map = prob[0, 0].cpu().detach().numpy() 
    # gt = labs[0] if labs.ndim == 3 else labs  # be safe
    # labs = labs[0]                                                       # [H,W]
    # output_map = prob[0, 0, :, :].cpu().detach().numpy()                 # [H,W] prob

    # input_img.to('cpu')


    # input_img = input_img[0].transpose(0,-1).cpu().detach().numpy()
    # labs = labs[0]
    # output = output[0,0,:,:].cpu().detach().numpy()

    ################################################################################
    # apply horizontal + vertical flip to labels and outputs
    # labs = np.fliplr(labs)
    # labs = np.flipud(labs)

    # output = np.fliplr(output)
    # output = np.flipud(output)

    # predict_save = np.fliplr(predict_save)
    # predict_save = np.flipud(predict_save)
    ################################################################################

    # if(True):
    #     pickle.dump({
    #         'input':input_img,
    #         'output':(output>=0.5)*1.0,            
    #         'ground_truth':labs,
    #         'dice':dice_pred_tmp,
    #         'iou':iou_tmp
    #     },
    #     open(vis_save_path+'.p','wb'))

    # pred_vis_path = os.path.join(vis_save_path, 'predicted_masks')
    # os.makedirs(pred_vis_path, exist_ok=True)

    # plt.savefig(pred_vis_path+'_predict'+model_type+'.png',dpi=300)

    fname = os.path.splitext(os.path.basename(vis_save_path))[0]

    # # --- 1) Save predicted mask only ---
    # mask_file = os.path.join(mask_dir, f"{fname}_mask_{model_type}.png")
    # plt.figure(figsize=(5,5))
    # plt.imshow((output >= 0.5) * 1.0, cmap="gray")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(mask_file, dpi=300, bbox_inches="tight", pad_inches=0)
    # plt.close()

    # # --- 2) Save side-by-side figure ---
    # side_file = os.path.join(side_dir, f"{fname}_side_{model_type}.png")
    # plt.figure(figsize=(12,4))

    # plt.subplot(1,3,1)
    # plt.imshow(input_img)
    # plt.axis("off")
    # plt.title("Input")

    # plt.subplot(1,3,2)
    # plt.imshow(labs, cmap="gray")
    # plt.axis("off")
    # plt.title("Ground Truth")

    # plt.subplot(1,3,3)
    # plt.imshow((output >= 0.5) * 1.0, cmap="gray")
    # plt.axis("off")
    # plt.title("Prediction")

    # plt.tight_layout()
    # plt.savefig(side_file, dpi=300)
    # plt.close()

    # --- 1) Save predicted mask only (pixel-perfect) ---
    mask_file = os.path.join(mask_dir, f"{fname}.png")
    # print(f"pred file path {mask_file}")
    # mask_file = os.path.join(mask_dir, f"{vis_path}.png")
    pred_mask = (output_map >= 0.5).astype(np.uint8) * 255  # binary mask → 0/255
    cv2.imwrite(mask_file, pred_mask)  # save exact resolution (no scaling)



    # Optional overlay visualization (nice for inspection)
    # overlay = cv2.addWeighted(
    #     (input_img * 255).astype(np.uint8), 0.7,
    #     cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR), 0.3, 0
    # )
    # cv2.imwrite(os.path.join(mask_dir, f"{fname}_overlay_{model_type}.png"), overlay)

    # --- 2) Save side-by-side figure (high-res + no interpolation) ---
    # side_file = os.path.join(side_dir, f"{fname}_side_{model_type}.png")

    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 3, 1)
    # plt.imshow(input_img_np)
    # plt.axis("off")
    # plt.title("Input")

    # plt.subplot(1, 3, 2)
    # plt.imshow(gt, cmap="gray", interpolation='nearest')
    # plt.axis("off")
    # plt.title("Ground Truth")

    # plt.subplot(1, 3, 3)
    # plt.imshow(pred_mask, cmap="gray", interpolation='nearest')
    # plt.axis("off")
    # plt.title("Prediction")

    # plt.tight_layout()
    # plt.savefig(side_file, dpi=600, bbox_inches="tight", pad_inches=0)
    # plt.close()


    # if(False):
        
        # plt.figure(figsize=(10,3.3))
        # plt.subplot(1,3,1)
        # plt.imshow(input_img)
        # plt.subplot(1,3,2)
        # plt.imshow(labs,cmap='gray')
        # plt.subplot(1,3,3)
        # plt.imshow((output>=0.5)*1.0,cmap='gray')    
        # plt.suptitle(f'Dice score : {np.round(dice_pred_tmp,3)}\nIoU : {np.round(iou_tmp,3)}')
        # plt.tight_layout()
        # plt.savefig(vis_save_path)
        # plt.close()

    # ---------- NEW: Grad-CAM overlay (2D) ----------
    if gradcam is not None:
        try:
            # input_img is [B,C,H,W]; send CUDA tensor
            cam = gradcam(input_img.cuda(), retain_graph=False)  # numpy [H,W] in [0,1]
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            cam = None

        if cam is not None:
            cam_uint8 = (cam * 255).astype(np.uint8)
            cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # BGR

            # convert input image to uint8
            img_uint8 = (input_img_np * 255).astype(np.uint8)
            if img_uint8.shape[2] == 1:
                img_uint8 = cv2.cvtColor(img_uint8[:, :, 0], cv2.COLOR_GRAY2BGR)
            else:
                # input is RGB -> convert to BGR for OpenCV
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            heatmap_resized = cv2.resize(
                cam_color,
                (img_uint8.shape[1], img_uint8.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            img_uint8 = cv2.rotate(img_uint8, cv2.ROTATE_90_CLOCKWISE)
            img_uint8 = cv2.flip(img_uint8, 1) 
            overlay = cv2.addWeighted(img_uint8, 0.6, heatmap_resized, 0.4, 0)


            overlay_file = os.path.join(cam_dir, f"{fname}_cam_overlay_{model_type}.png")
            print("Saving Grad-CAM overlay to:", overlay_file)
            cv2.imwrite(overlay_file, overlay)


    return dice_pred_tmp, iou_tmp, output_map 
    # return dice_pred_tmp, iou_tmp





if __name__ == '__main__':
    

    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name =="GlaS_exp1":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="GlaS_exp2":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="GlaS_exp3":
        test_num = 80
        model_type = config.model_name
        model_path = "./GlaS_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    
    elif config.task_name =="ISIC18_exp1":
        test_num = 518
        model_type = config.model_name
        model_path = "./ISIC18_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"


    ################################################################################################################
    elif config.task_name =="ISIC18_UNET":
        test_num = 1000
        model_type = config.model_name
        model_path = "./ISIC18_UNET/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="ISIC18_mod":
        test_num = 996
        model_type = config.model_name
        model_path = "./ISIC18_UNET/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"   

    elif config.task_name =="ISIC17":
        test_num = 600
        model_type = config.model_name
        model_path = "./ISIC17/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="MoNuSeg":
        test_num = 14
        model_type = config.model_name
        model_path = "./MoNuSeg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="Kvasir-Seg":
        test_num = 99
        model_type = config.model_name
        model_path = "./Kvasir-Seg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="ClinicDB":
        test_num = 65
        model_type = config.model_name
        model_path = "./Kvasir-Seg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="ColonDB":
        test_num = 400
        model_type = config.model_name
        model_path = "./Kvasir-Seg/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="CVC-ClinicDB":
        test_num = 62
        model_type = config.model_name
        model_path = "./CVC-ClinicDB/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="BUSI":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

# dwe
    elif config.task_name =="BUSI_80-20":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI_80-20/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="BUSI_80-20_mod":
        test_num = 128
        model_type = config.model_name
        model_path = "./BUSI_80-20/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    


    elif config.task_name =="CVC_ClinicDB_80-20":
        test_num = 123
        model_type = config.model_name
        model_path = "./CVC_ClinicDB_80-20/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="Glas":
        test_num = 130
        model_type = config.model_name
        model_path = "./Glas/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="Glas_80-20":
        test_num = 33
        model_type = config.model_name
        model_path = "./Glas_80-20/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="TNBC_80-20":
        test_num = 10
        model_type = config.model_name
        model_path = "./TNBC_80-20/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    



    #################################################################################################################


    elif config.task_name =="BUSI_UNET":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI_UNET/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"    

    elif config.task_name =="ISIC18_exp2":
        test_num = 518
        model_type = config.model_name
        model_path = "./ISIC18_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="ISIC18_exp3":
        test_num = 518
        model_type = config.model_name
        model_path = "./ISIC18_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    
    elif config.task_name =="Clinic_exp1":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Clinic_exp2":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Clinic_exp3":
        test_num = 122
        model_type = config.model_name
        model_path = "./Clinic_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    

    elif config.task_name =="BUSI_exp1":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="BUSI_exp2":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="BUSI_exp3":
        test_num = 130
        model_type = config.model_name
        model_path = "./BUSI_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"
    
    

    elif config.task_name =="Covid_exp1":
        test_num = 20
        model_type = config.model_name
        model_path = "./Covid_exp1/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Covid_exp2":
        test_num = 20
        model_type = config.model_name
        model_path = "./Covid_exp2/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"

    elif config.task_name =="Covid_exp3":
        test_num = 20
        model_type = config.model_name
        model_path = "./Covid_exp3/"+model_type+"/"+test_session+"/models/best_model-"+model_type+".pth.tar"


    save_path  = config.task_name +'/'+ config.model_name +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    vis_path = save_path + 'visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)


    # NEW: folder only for predicted images
    pred_img_path = os.path.join(save_path, "predicted_images")
    os.makedirs(pred_img_path, exist_ok=True)

    # NEW: folders for predicted outputs
    # pred_img_path = os.path.join(save_path, "predicted_images")
    mask_dir = os.path.join(pred_img_path, "masks")
    side_dir = os.path.join(pred_img_path, "side_by_side")

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(side_dir, exist_ok=True)

    cam_dir = os.path.join(pred_img_path, "cam")
    os.makedirs(cam_dir, exist_ok=True)

    checkpoint = torch.load(model_path, map_location='cuda')
    print("=> loading model from {}".format(model_path))
    state_dict = checkpoint['state_dict']

    clean_state_dict = {
                        k: v for k, v in state_dict.items()
                        if "total_ops" not in k and "total_params" not in k
                        }

    # load with strict=False so extra keys don’t break
    

    fp = open(save_path+'test.result','a')
    fp.write(str(datetime.now())+'\n')

    
    if model_type == 'ACC_UNet':
        config_vit = config.get_CTranS_config()   
        model = ACC_UNet(n_channels=config.n_channels,n_classes=config.n_labels,n_filts=config.n_filts)

    elif model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'UNet_base':
        config_vit = config.get_CTranS_config()   
        model = UNet_base(n_channels=config.n_channels,n_classes=config.n_labels)
        
    elif model_type == 'SwinUnet':            
        model = SwinUnet()
        model.load_from()

    elif model_type == 'SMESwinUnet':            
        model = SMESwinUnet(n_channels=config.n_channels,n_classes=config.n_labels)
        model.load_from()

    elif model_type == 'UNeXt':
        model = UNext(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4  

    elif model_type == 'U-KAN':
        model = UKAN(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_InceptionNext_MLFC':
        model = UNext_InceptionNext_MLFC(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4  

    elif model_type == 'UNext_CMRF':
        model = UNext_CMRF(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4

    elif model_type == 'UNext_CMRF_enc_dec':
        model = UNext_CMRF_enc_dec(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_enc_MLFC':
        model = UNext_CMRF_enc_MLFC(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_enc_dec_MLFC':
        model = UNext_CMRF_enc_dec_MLFC(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_enc_CSSE':
        model = UNext_CMRF_enc_CSSE(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_dense_skip':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        # model = UNext_CMRF_PP_UNetPP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_Dense_Skip(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'TransUNet':
        model = TransUNet(n_channels=config.n_channels, n_classes=config.n_labels)
        # good defaults for ViT-based models:
        # lr = 1e-4
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    # lr=lr, weight_decay=0.01)

    elif model_type == 'UNext_CMRF_GAB':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GAB(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_GAB_wavelet':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GAB_Wavelet(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_GAB_wavelet_OD':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GAB_Wavelet_OD(n_channels=config.n_channels, n_classes=config.n_labels)
    
    elif model_type == 'UNext_CMRF_GS':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GS(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_GS_Wavelet':
        model = UNext_CMRF_GS_Wavelet(n_channels=config.n_channels, n_classes=config.n_labels)
    
    elif model_type == 'UNext_CMRF_GS_Wavelet_hd':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GS_Wavelet_hd(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_BS_GS_Wavelet':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_BS_GS_Wavelet(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_GS_Wavelet_OD':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GS_Wavelet_OD(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_BSRB_GS_Wavelet':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_BSRB_GS_Wavelet(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_BSRB_GS':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_BSRB_GS(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_GS_Wavelet_rKAN':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GS_Wavelet_rKAN(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type.split('_')[0] == 'MultiResUnet1':          
        model = MultiResUnet(n_channels=config.n_channels,n_classes=config.n_labels,nfilt=int(model_type.split('_')[1]), alpha=float(model_type.split('_')[2]))
    
    # elif model_type == 'Segmamba_hybrid_gsc_rm_fkan':
    #     model = Segmamba_hybrid_gsc_rm_fkan(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    elif model_type == 'Segmamba_hybrid_gsc_KAN_PE_ds':
        model = Segmamba_hybrid_gsc_KAN_PE_ds(
            in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384], spatial_dims=3,)
        lr = 1e-4

    # elif model_type == 'Segmamba':
    #     model = SegMamba(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()

    # FLOPS CALCULATION
    dummy_input = torch.randn(1, config.n_channels, config.img_size, config.img_size).cuda()
    # macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    # model_params = params / 1e6
    # model_gflops = macs / 1e9

    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    # model.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(clean_state_dict, strict=False)
    print(" Clean model weights loaded!")
    print(model_type)
    print('Model loaded !')

    dummy_input = torch.randn(1, config.n_channels, config.img_size, config.img_size).cuda()
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    model_params = params / 1e6
    model_gflops = macs / 1e9
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    accuracy_meter = AverageMeter()
    sensitivity_meter = AverageMeter()
    specificity_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    gpu_time_meter = AverageMeter()
    f1_meter = AverageMeter()


    # --------- Grad-CAM init (SegMamba or others) ----------
    gradcam = None
    try:
        # Use underlying module if DataParallel
        use_model_for_cam = model.module if isinstance(model, nn.DataParallel) else model

        if model_type == 'Segmamba_hybrid_gsc_KAN_PE_ds':
            # Good candidate layer: decoder1 (high-res features before final head)
            target_layer = use_model_for_cam.decoder1
            print("Using Grad-CAM target layer: decoder1")
        else:
            target_layer = None

        if target_layer is not None:
            gradcam = SegGradCAM(use_model_for_cam, target_layer)
            print("Grad-CAM initialized successfully.")
        else:
            print("Grad-CAM not initialized (no target_layer for this model_type).")
    except Exception as e:
        print(f"Grad-CAM initialization failed: {e}")
        gradcam = None


    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            # arr = arr.astype(np.float32())
            arr = arr.astype(np.float32)
            lab=test_label.data.numpy()
            # img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            img_lab = lab.reshape(config.img_size, config.img_size) * 255
            ###########fig, ax = plt.subplots()
            ###########plt.imshow(img_lab, cmap='gray')
            ###########plt.axis("off")
            height, width = config.img_size, config.img_size
            ###########fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            ###########plt.gca().xaxis.set_major_locator(plt.NullLocator())
            ###########plt.gca().yaxis.set_major_locator(plt.NullLocator())
            ###########plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            ###########plt.margins(0, 0)
            ###########plt.savefig(vis_path+str(i)+"_lab.jpg", dpi=300)
            ###########plt.close()
            input_img = torch.from_numpy(arr)
            # dice_pred_t,iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
            #                                               vis_path+str(i)+'.png',
            #                                    dice_pred=dice_pred, dice_ens=dice_ens)
            # dice_pred_t, iou_pred_t, output = vis_and_save_heatmap(
            #                                         model, input_img, None, lab,
            #                                         vis_path + str(i) + '.png',
            #                                         dice_pred=dice_pred,
            #                                         dice_ens=dice_ens
            #                                     )

            # dice_pred_t, iou_pred_t, output = vis_and_save_heatmap(
            #                                     model, input_img, None, lab,
            #                                     vis_path + str(i),          # pickles still go here
            #                                     dice_pred=dice_pred,
            #                                     dice_ens=dice_ens,
            #                                     pred_img_path=pred_img_path # new folder for .pngs
            #                                 )

            # dice_pred_t, iou_pred_t, output = vis_and_save_heatmap(
            #                                         model, input_img, None, lab,
            #                                         vis_path + str(i),
            #                                         dice_pred=dice_pred,
            #                                         dice_ens=dice_ens,
            #                                         mask_dir=mask_dir,
            #                                         side_dir=side_dir
            #                                     )

            original_filename = os.path.splitext(names[0])[0]  # e.g. "ISIC_0036347"

            # dice_pred_t, iou_pred_t, output = vis_and_save_heatmap(
            #     model, input_img, None, lab,
            #     vis_path + original_filename,    # ✅ use real filename here
            #     dice_pred=dice_pred,
            #     dice_ens=dice_ens,
            #     mask_dir=mask_dir,
            #     side_dir=side_dir
            # )

            dice_pred_t, iou_pred_t, output = vis_and_save_heatmap(
                model, input_img, None, lab,
                vis_path + original_filename,
                dice_pred=dice_pred,
                dice_ens=dice_ens,
                mask_dir=mask_dir,
                side_dir=side_dir,
                cam_dir=cam_dir,
                gradcam=gradcam
            )



            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            # output_bin = (output > 0.5).float().cpu().numpy()
            output_bin = (output > 0.5).astype(np.float32)
            target_bin = lab

            TP = ((output_bin == 1) & (target_bin == 1)).sum()
            TN = ((output_bin == 0) & (target_bin == 0)).sum()
            FP = ((output_bin == 1) & (target_bin == 0)).sum()
            FN = ((output_bin == 0) & (target_bin == 1)).sum()

            eps = 1e-7
            sensitivity = TP / (TP + FN + eps)
            specificity = TN / (TN + FP + eps)
            accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)

            f1 = (2 * precision * recall) / (precision + recall + eps)

            sensitivity_meter.update(sensitivity)
            specificity_meter.update(specificity)
            accuracy_meter.update(accuracy)
            precision_meter.update(precision)
            recall_meter.update(recall)
            f1_meter.update(f1) 

            torch.cuda.empty_cache()
            pbar.update()
    print("Test completed!")
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)
    print(f"Precision: {precision_meter.avg:.4f}")
    print(f"Recall: {recall_meter.avg:.4f}")
    print(f"F1: {f1_meter.avg:.4f}")
    print(f"Sensitivity: {sensitivity_meter.avg * 100:.2f}%")
    print(f"Specificity: {specificity_meter.avg * 100:.2f}%")
    print(f"Accuracy: {accuracy_meter.avg * 100:.2f}%")
    print(f"Params: {model_params:.2f} M")
    print(f"GFLOPs: {model_gflops:.2f} G")
    print(f"Avg GPU Time/Image: {gpu_time_meter.avg:.4f} sec")


    fp.write(f"Precision: {precision_meter.avg:.4f}\n")
    fp.write(f"Recall: {recall_meter.avg:.4f}\n")
    fp.write(f"F1: {f1_meter.avg:.4f}\n")
    fp.write(f"Sensitivity: {sensitivity_meter.avg * 100:.2f}%\n")
    fp.write(f"Specificity: {specificity_meter.avg * 100:.2f}%\n")
    fp.write(f"Accuracy: {accuracy_meter.avg * 100:.2f}%\n")
    fp.write(f"Params (M): {model_params:.2f}\n")
    fp.write(f"GFLOPs: {model_gflops:.2f}\n")
    fp.write(f"Avg GPU Time (s): {gpu_time_meter.avg:.4f}\n")
    
    
    fp.write(f"dice_pred : {dice_pred/test_num}\n")
    fp.write(f"iou_pred : {iou_pred/test_num}\n")
    fp.close()

    # SAVE RESULTS TO CSV
    metrics_dict = {
    'IoU': [float(iou_pred/test_num)],
    'Dice': [float(dice_pred/test_num)],
    'Precision': [float(precision_meter.avg)],
    'Recall': [float(recall_meter.avg)],
    'F1': [float(f1_meter.avg)], 
    'Sensitivity (%)': [float(sensitivity_meter.avg * 100)],
    'Specificity (%)': [float(specificity_meter.avg * 100)],
    'Accuracy (%)': [float(accuracy_meter.avg * 100)],
    'Params (M)': [float(model_params)],
    'GFLOPs': [float(model_gflops)],
    'Avg GPU Time (s)': [float(gpu_time_meter.avg)],
    }

    df = pd.DataFrame(metrics_dict)
    csv_path = os.path.join(save_path, 'metrics_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved metrics to {csv_path}")