"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # change this as needed

use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 32            # change this to train larger ACC-UNet model
cosineLR = True         # whether use cosineLR or not
n_channels = 3
n_labels = 1
epochs = 10000
# epochs = 400

print_frequency = 200
save_frequency = 100
vis_frequency = 100
early_stopping_patience = 100
# early_stopping_patience = 30
# early_stopping_patience = 400

pretrain = False
resume_checkpoint = False  # or False


# task_name = 'GlaS_exp1'
# task_name = 'ISIC18_exp1'
#task_name = 'Clinic_exp1'
#task_name = 'Covid_exp1'
#task_name = 'BUSI_exp1'


# task_name = 'ISIC18_UNET'
# task_name = 'ISIC18_mod'
# task_name = 'ISIC17'
# task_name = 'MoNuSeg'
# task_name = 'CVC-ClinicDB'
# task_name = 'Kvasir-Seg'
# task_name = 'BUSI'
# task_name = 'Glas'
# task_name = 'ClinicDB'
# task_name = 'ColonDB'

task_name = 'BUSI_80-20'
# task_name = 'BUSI_80-20_mod'

# task_name = 'CVC_ClinicDB_80-20'
# task_name = 'Glas_80-20'
# task_name = 'TNBC_80-20'
# task_name = 'STARE'
# task_name = 'DRIVE'
# task_name = 'BUSI_UNET'

learning_rate = 1e-3
# learning_rate = 0.0001
# batch_size = 32
# batch_size = 8
batch_size = 2

# model_name = 'ACC_UNet'
# model_name = 'SwinUnet'
# model_name = 'SMESwinUnet'
# model_name = 'UCTransNet'
# model_name = 'UNet_base'
# model_name = 'UNet_base_proto'
# model_name = 'MultiResUnet1_32_1.67'


# model_name = 'UNeXt'
# model_name = 'UNext_CMRF_GS_Wavelet'  # CMRF encoder + Global Semnantic + SIM augmentation + wavelet 

# model_name = 'UNext_CMRF_GS_Wavelet_rKAN'
# model_name = 'UNext_InceptionNext_MLFC_fKAN'

# model_name = 'Segmamba'
# model_name = 'Segmamba_hybrid'
# model_name = 'Segmamba_hybrid_gsc'
# model_name = 'Segmamba_hybrid_gsc_CA'
# model_name = 'Segmamba_hybrid_gsc_SWAttn'
# model_name = 'Segmamba_hybrid_gsc_VSS'
# model_name = 'Segmamba_hybrid_gsc_KAN_PE'
# model_name = 'Segmamba_hybrid_gsc_KAN_PE_ds'
model_name = 'Segmamba_hybrid_gsc_KAN_PE_ds_flip'
# model_name = 'segmamba_hybrid_gsc_KAN_PE_EffKan'
# model_name = 'Segmamba_hybrid_gsc_rm_fkan'
# model_name = 'Segmamba_hybrid_gsc_KAN_PE_rm_fkan'
# model_name = 'Segmamba_hybrid_gsc_KAN_PE_rm_fkan_ds'
# model_name = 'Segmamba_hybrid_gsc_ds'
# model_name = 'TransUNet_fJNB'
# model_name = 'SegViT_fKAN'

# model_name = 'UNext_InceptionNext_MLFC'
# model_name = 'UNext_CMRF'   # CMRF encoder
# model_name = 'UNext_CMRF_enc_dec'  # CMRF encoder + decoder
# model_name = 'UNext_CMRF_enc_MLFC'  # CMRF encoder + MLFC fusion
# model_name = 'UNext_CMRF_enc_dec_MLFC'  # CMRF encoder + decoder + MLFC fusion
# model_name = 'UNext_CMRF_enc_CSSE'
# model_name = 'UNext_CMRF_PP'
# model_name = 'TransUNet'  # TransUNet model
# model_name = 'UNext_CMRF_GAB'  # CMRF encoder + GAB fusion
# model_name = 'UNext_CMRF_GAB_wavelet'  # CMRF encoder + GAB fusion + wavelet
# model_name = 'UNext_CMRF_GAB_wavelet_OD'  # CMRF encoder + GAB fusion + wavelet + ODConv
# model_name = 'UNext_CMRF_BS_GS_Wavelet'  # CMRF encoder + BSConv + GS + SIM augmentation + wavelet
# model_name = 'UNext_CMRF_BSRB_GS_Wavelet'  # CMRF encoder + BSRB + GS + SIM augmentation + wavelet
# model_name = 'UNext_CMRF_BSRB_GS'  # CMRF encoder + BSRB + GS + SIM augmentation
# model_name = 'UNext_CMRF_GS_Wavelet_OD'  # CMRF encoder + Global Semnantic + SIM augmentation + wavelet + ODConv
# model_name = 'UNext_CMRF_GS_Wavelet_hd'  
# model_name = 'UNext_CMRF_GS'  # CMRF encoder + Global Semnantic + SIM augmentation
# model_name = 'UNext_CMRF_dense_skip'  # CMRF encoder + dense skip connection
# model_name = 'U-KAN'

# if model_name == 'SwinUnet' or model_name == 'UCTransNet' or model_name == 'Segmamba' or model_name == 'Segmamba_hybrid'or model_name == 'Segmamba_hybrid_gsc' or model_name == 'Segmamba_hybrid_gsc_CA':
#     img_size = 224
# else :
#     img_size = 256

models_224 = {
    'SwinUnet', 'UCTransNet', 'Segmamba', 'Segmamba_hybrid',
    'Segmamba_hybrid_gsc', 'Segmamba_hybrid_gsc_CA', 'Segmamba_hybrid_gsc_SWAttn',
    'Segmamba_hybrid_gsc_VSS', 'Segmamba_hybrid_gsc_KAN_PE', 'Segmamba_hybrid_gsc_rm_fkan',
    'Segmamba_hybrid_gsc_KAN_PE_rm_fkan', 'Segmamba_hybrid_gsc_ds', 'Segmamba_hybrid_gsc_KAN_PE_rm_fkan_ds',
    'Segmamba_hybrid_gsc_KAN_PE_ds', 'segmamba_hybrid_gsc_KAN_PE_EffKan', 'Segmamba_hybrid_gsc_KAN_PE_ds_flip'
}

img_size = 224 if model_name in models_224 else 256

# img_size = 224
test_session = "session2"         #


# train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
# val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
# test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
# session_name       = 'session'  #time.strftime('%m.%d_%Hh%M')
# save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
# model_path         = save_path + 'models/'
# tensorboard_folder = save_path + 'tensorboard_logs/'
# logger_path        = save_path + session_name + ".log"
# visualize_path     = save_path + 'visualize_val/'

# dataset_path = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2'
# train_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2/train'
# val_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2/val'
# test_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2/test'

# ISIC 18 MOD
# dataset_path = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic_mod'
# train_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic_mod/train'
# val_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic_mod/val'
# test_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic_mod/test'

# ISIC 17
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/ISIC_2017/Dataset_ISIC_2017_Formatted'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/ISIC_2017/Dataset_ISIC_2017_Formatted/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/ISIC_2017/Dataset_ISIC_2017_Formatted/val'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/ISIC_2017/Dataset_ISIC_2017_Formatted/test'


# BUSI_80-20
dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20'
train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/train'
val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/test'
test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/test'


# BUSI_80-20_mod
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/test_mod'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_BUSI_80_20/test_mod'


# Glas 80-20
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_GlaS_80_20'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_GlaS_80_20/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_GlaS_80_20/test'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_GlaS_80_20/test'

# TNBC 80-20
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_TNBC_80_20'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_TNBC_80_20/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_TNBC_80_20/val'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Dataset_TNBC_80_20/val'


# MoNuSeg

# dataset_path = '/content/drive/MyDrive/Akanksha/Monuseg_Dataset'
# train_dataset = '/content/drive/MyDrive/Akanksha/Monuseg_Dataset/train'
# val_dataset = '/content/drive/MyDrive/Akanksha/Monuseg_Dataset/val'
# test_dataset = '/content/drive/MyDrive/Akanksha/Monuseg_Dataset/test'

# CVC-ClinicDB

# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/CVC-ClinicDB'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/CVC-ClinicDB/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/CVC-ClinicDB/val'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/CVC-ClinicDB/test'



# ISIC 18
# dataset_path = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1'
# train_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/train'
# val_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/val'
# test_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/test'

# Glas
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/Glas'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Glas/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Glas/test'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/Glas/test'

# BUSI
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/BUSI_ACC/'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/BUSI_ACC/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/BUSI_ACC/val'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/BUSI_ACC/test'

# Kvasir-Seg

# dataset_path = '/content/drive/MyDrive/Akanksha/PFNET_2_2_8_2/PFNet/data/Kvasir-SEG'
# train_dataset = '/content/drive/MyDrive/Akanksha/PFNET_2_2_8_2/PFNet/data/Kvasir-SEG/train'
# val_dataset = '/content/drive/MyDrive/Akanksha/PFNET_2_2_8_2/PFNet/data/Kvasir-SEG/val'

# test_dataset = '/content/drive/MyDrive/Akanksha/PFNET_2_2_8_2/PFNet/data/Kvasir-SEG/test'
    # ClinicDB
    # test_dataset = '/content/drive/MyDrive/Akanksha/PFNET_2_2_8_2/PFNet/data/CVC-ClinicDB'
    # ColonDB
    # test_dataset = '/content/drive/MyDrive/Akanksha/PFNET_2_2_8_2/PFNet/data/CVC-ColonDB'



# CVC_ClinicDB_80-20
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/CVC/CVC-ClinicDB_80_20'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/CVC/CVC-ClinicDB_80_20/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/CVC/CVC-ClinicDB_80_20/test'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/CVC/CVC-ClinicDB_80_20/test'

# STARE
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/STARE_10_10'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/STARE_10_10/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/STARE_10_10/test'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/STARE_10_10/test'

# DRIVE
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/DRIVE'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/DRIVE/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/DRIVE/test'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/DRIVE/test'



session_name       = 'session2'  #time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + ".log"
# logger_path = '/content/drive/MyDrive/Prashant/ACC-UNet-Unext/Experiments/train_logs/session_name/.log'
visualize_path     = save_path + 'visualize_val/'



##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config




# used in testing phase, copy the session name in training phase
