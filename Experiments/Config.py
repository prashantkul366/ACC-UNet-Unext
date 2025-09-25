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
epochs = 1000
# epochs = 400
# img_size = 224
img_size = 256
print_frequency = 10
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


task_name = 'ISIC18_UNET'
# task_name = 'BUSI_UNET'

learning_rate = 1e-3
# learning_rate = 0.0001
# batch_size = 32
batch_size = 8

# model_name = 'ACC_UNet'
# model_name = 'SwinUnet'
# model_name = 'SMESwinUnet'
# model_name = 'UCTransNet'
# model_name = 'UNet_base'
# model_name = 'UNet_base_proto'
# model_name = 'MultiResUnet1_32_1.67'
# model_name = 'UNeXt'
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
model_name = 'UNext_CMRF_BS_GS_Wavelet'  # CMRF encoder + BSConv + GS + SIM augmentation + wavelet
# model_name = 'UNext_CMRF_GS_Wavelet_OD'  # CMRF encoder + Global Semnantic + SIM augmentation + wavelet + ODConv
# model_name = 'UNext_CMRF_GS_Wavelet_hd'  
# model_name = 'UNext_CMRF_GS'  # CMRF encoder + Global Semnantic + SIM augmentation
# model_name = 'UNext_CMRF_GS_Wavelet'  # CMRF encoder + Global Semnantic + SIM augmentation + wavelet 
# model_name = 'UNext_CMRF_dense_skip'  # CMRF encoder + dense skip connection
# model_name = 'U-KAN'

test_session = "session1"         #


# train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
# val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
# test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
# session_name       = 'session'  #time.strftime('%m.%d_%Hh%M')
# save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
# model_path         = save_path + 'models/'
# tensorboard_folder = save_path + 'tensorboard_logs/'
# logger_path        = save_path + session_name + ".log"
# visualize_path     = save_path + 'visualize_val/'

dataset_path = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2'
train_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2/train'
val_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2/val'
test_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic2/test'


# MoNuSeg

# dataset_path = '/content/drive/MyDrive/Akanksha/Monuseg_Dataset'
# train_dataset = '/content/drive/MyDrive/Akanksha/Monuseg_Dataset/train'
# val_dataset = '/content/drive/MyDrive/Akanksha/Monuseg_Dataset/val'
# test_dataset = '/content/drive/MyDrive/Akanksha/Monuseg_Dataset/test'


# ISIC 18
# dataset_path = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1'
# train_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/train'
# val_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/val'
# test_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/test'

# Glas
# dataset_path = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1'
# train_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/train'
# val_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/val'
# test_dataset = '/content/drive/MyDrive/Amit-Paper3/UNeXt-pytorch/inputs/isic1/test'

# BUSI
# dataset_path = '/content/drive/MyDrive/Prashant/research_datasets/BUSI_ACC/'
# train_dataset = '/content/drive/MyDrive/Prashant/research_datasets/BUSI_ACC/train'
# val_dataset = '/content/drive/MyDrive/Prashant/research_datasets/BUSI_ACC/val'
# test_dataset = '/content/drive/MyDrive/Prashant/research_datasets/BUSI_ACC/test'


session_name       = 'session1'  #time.strftime('%m.%d_%Hh%M')
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
