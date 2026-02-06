"""
Our experimental codes are based on 
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import torch.optim
from tensorboardX import SummaryWriter
import os
import numpy as np
import random
from torch.backends import cudnn
from torch.optim import lr_scheduler as torch_lr_scheduler
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet
from nets.UNet_base import UNet_base
from nets.UNext import UNext
from nets.archs.u_kan import UKAN
from nets.ACC_UNet import ACC_UNet


# from nets.SMESwinUnet import SMESwinUnet
# from nets.UCTransNet import UCTransNet

##################### NEW ARCHS ######################

# from nets.archs.archs_InceptionNext_MLFC import UNext_InceptionNext_MLFC
# from nets.archs.UNext_CMRF import UNext_CMRF
# from nets.archs.UNext_CMRF_enc_dec import UNext_CMRF_enc_dec
# from nets.archs.UNext_CMRF_enc_MLFC import UNext_CMRF_enc_MLFC
# from nets.archs.UNext_CMRF_enc_dec_MLFC import UNext_CMRF_enc_dec_MLFC
# from nets.archs.UNext_CMRF_enc_CSSE import UNext_CMRF_enc_CSSE
# from nets.archs.UNext_CMRF_PP import UNext_CMRF_PP_UNetPP
# from nets.archs.UNext_CMRF_dense_skip import UNext_CMRF_Dense_Skip

# from nets.archs.UNext_CMRF_GAB import UNext_CMRF_GAB
# from nets.archs.UNext_CMRF_GS import UNext_CMRF_GS

# from nets.TransUNet import TransUNet


# from nets.archs.UNext_CMRF_GS_wavelet import UNext_CMRF_GS_Wavelet
# from nets.archs.UNext_CMRF_GS_wavelet_OD import UNext_CMRF_GS_Wavelet_OD

# from nets.archs.UNext_CMRF_GAB_wavelet import UNext_CMRF_GAB_Wavelet
# from nets.archs.UNext_CMRF_GAB_wavelet_OD import UNext_CMRF_GAB_Wavelet_OD
# from nets.archs.UNext_CMRF_GS_wavelet_hd import UNext_CMRF_GS_Wavelet_hd

# from nets.archs.UNext_CMRF_BS_GS_wavelet import UNext_CMRF_BS_GS_Wavelet
# from nets.archs.UNext_CMRF_BSRB_GS_wavelet import UNext_CMRF_BSRB_GS_Wavelet

# from nets.archs.UNext_CMRF_BSRB_GS import UNext_CMRF_BSRB_GS

# from nets.archs.UNext_CMRF_GS_wavelet_rkan import UNext_CMRF_GS_Wavelet_rKAN
# from nets.archs.archs_InceptionNext_MLFC_fKAN import UNext_InceptionNext_MLFC_fKAN

######################################################
# from nets.segmamba import SegMamba
# from nets.segmamba_hybrid import SegMamba as SegMamba_hybrid
# from nets.segmamba_hybrid_gsc import SegMamba as SegMamba_hybrid_gsc
# from nets.segmamba_hybrid_gsc_CA import SegMamba as Segmamba_hybrid_gsc_CA
# from nets.segmamba_hybrid_gsc_SWAttn import SegMamba as Segmamba_hybrid_gsc_SWAttn
# from nets.segmamba_hybrid_gsc_vss import SegMamba as Segmamba_hybrid_gsc_VSS
# from nets.segmamba_hybrid_gsc_KAN_PE import SegMamba as Segmamba_hybrid_gsc_KAN_PE
# from nets.segmamba_hybrid_gsc_rm_fkan import SegMamba as Segmamba_hybrid_gsc_rm_fkan
# from nets.segmamba_hybrid_gsc_KAN_PE_rm_fkan import SegMamba as Segmamba_hybrid_gsc_KAN_PE_rm_fkan
# from nets.segmamba_hybrid_gsc_ds import SegMamba as Segmamba_hybrid_gsc_ds
# from nets.segmamba_hybrid_gsc_KAN_PE_rm_fkan_ds import SegMamba as Segmamba_hybrid_gsc_KAN_PE_rm_fkan_ds


from nets.segmamba_hybrid_gsc_KAN_PE_ds import SegMamba as Segmamba_hybrid_gsc_KAN_PE_ds
# from nets.segmamba_hybrid_gsc_MLP_PE_ds import SegMamba as Segmamba_hybrid_gsc_MLP_PE_ds

# from nets.segmamba_hybrid_gsc_KAN_PE_ds_SPATIAL import SegMamba as Segmamba_hybrid_gsc_KAN_PE_ds_SPATIAL


# from nets.segmamba_hybrid_gsc_KAN_PE_EffKan import SegMamba as segmamba_hybrid_gsc_KAN_PE_EffKan
# from nets.segmamba_hybrid_gsc_KAN_PE_ds_flip import SegMamba as Segmamba_hybrid_gsc_KAN_PE_ds_flip
# from nets.TransUnet_fKAN import TransUNet_KAN_fJNB
# from nets.TransUNet_Vit_fKAN import TransUNet as TransUNet_KAN_fJNB
# from nets.seg_fViT import SegViT_fKAN

####################################################

from torch.utils.data import DataLoader
import logging
import json
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE, DSAdapterLoss, WeightedDiceBCEHausdorff
from utils import BinaryDiceBCE
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''
    logger.info('\t Saving to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    model = state['model']  # model type

    if best_model:
        filename = save_path + '/' + \
                   'best_model-{}.pth.tar'.format(model)
    else:
        filename = save_path + '/' + \
                   'model-{}-{:02d}.pth.tar'.format(model, epoch)
    torch.save(state, filename)

def worker_init_fn(worker_id):
    random.seed(config.seed + worker_id)

##################################################################################
#=================================================================================
#          Main Loop: load model,
#=================================================================================
##################################################################################
# def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True):
def main_loop(batch_size=config.batch_size, model_type='', tensorboard=True, resume=False):

    # Load train and val data
    
    train_tf= transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
    val_tf = ValGenerator(output_size=[config.img_size, config.img_size])
    train_dataset = ImageToImage2D(config.train_dataset, train_tf,image_size=config.img_size,)
    val_dataset = ImageToImage2D(config.val_dataset, val_tf,image_size=config.img_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              num_workers=8,
                              pin_memory=True)
    
    print("Training Data Loaded!!")
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True)

    print("Val Data Loaded!!")
    lr = config.learning_rate
    
    print("length of train dataset: ", len(train_dataset))
    print("length of val dataset: ", len(val_dataset))
    
    logger.info(model_type)
    logger.info('n_filts : ' + str(config.n_filts))

    if model_type == 'ACC_UNet':
        config_vit = config.get_CTranS_config()        
        model = ACC_UNet(n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'UCTransNet':
        config_vit = config.get_CTranS_config()
        logger.info('transformer head num: {}'.format(config_vit.transformer.num_heads))
        logger.info('transformer layers num: {}'.format(config_vit.transformer.num_layers))
        logger.info('transformer expand ratio: {}'.format(config_vit.expand_ratio))
        model = UCTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'UNet_base':
        config_vit = config.get_CTranS_config()        
        model = UNet_base(n_channels=config.n_channels,n_classes=config.n_labels)

    elif model_type == 'SMESwinUnet':
        config_vit = config.get_CTranS_config()        
        model = SMESwinUnet(n_channels=config.n_channels,n_classes=config.n_labels)
        model.load_from()
        lr = 5e-4

    elif model_type == 'SwinUnet':            
        model = SwinUnet()
        model.load_from()
        lr = 5e-4


    elif model_type.split('_')[0] == 'MultiResUnet1':          
        model = MultiResUnet(n_channels=config.n_channels,n_classes=config.n_labels,nfilt=int(model_type.split('_')[1]), alpha=float(model_type.split('_')[2]))        

    elif model_type == 'UNeXt':
        model = UNext(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4  

    elif model_type == 'UNext_InceptionNext_MLFC':
        # model = UNext_InceptionNext_MLFC(n_channels=config.n_channels, n_classes=config.n_labels)
        pass
        # lr = 1e-4  

    elif model_type == 'UNext_CMRF':
        model = UNext_CMRF(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4

    elif model_type == 'U-KAN':
        model = UKAN(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_enc_dec':
        model = UNext_CMRF_enc_dec(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4

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
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GS_Wavelet(n_channels=config.n_channels, n_classes=config.n_labels)
    
    elif model_type == 'UNext_CMRF_GS_Wavelet_hd':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GS_Wavelet_hd(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_GS_Wavelet_OD':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GS_Wavelet_OD(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_BS_GS_Wavelet':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_BS_GS_Wavelet(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_GS_Wavelet_rKAN':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_GS_Wavelet_rKAN(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_BSRB_GS_Wavelet':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_BSRB_GS_Wavelet(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'UNext_CMRF_BSRB_GS':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_CMRF_BSRB_GS(n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'TransUNet':
        model = TransUNet(n_channels=config.n_channels, n_classes=config.n_labels)
        # good defaults for ViT-based models:
        # lr = 1e-4
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    # lr=lr, weight_decay=0.01)

    elif model_type == 'UNext_InceptionNext_MLFC_fKAN':
        # model = UNext_CMRF_PP(n_channels=config.n_channels, n_classes=config.n_labels)
        model = UNext_InceptionNext_MLFC_fKAN(n_channels=config.n_channels, n_classes=config.n_labels)


    elif model_type == 'Segmamba':
        model = SegMamba(
            in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384], spatial_dims=3,)
        lr = 1e-4

    # elif model_type == 'Segmamba_hybrid':
    #     model = SegMamba_hybrid(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    # elif model_type == 'Segmamba_hybrid_gsc':
    #     model = SegMamba_hybrid_gsc(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    # elif model_type == 'Segmamba_hybrid_gsc_CA':
    #     model = Segmamba_hybrid_gsc_CA(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    # elif model_type == 'Segmamba_hybrid_gsc_SWAttn':
    #     model = Segmamba_hybrid_gsc_SWAttn(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    # elif model_type == 'Segmamba_hybrid_gsc_VSS':
    #     model = Segmamba_hybrid_gsc_VSS(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    # elif model_type == 'Segmamba_hybrid_gsc_KAN_PE':
    #     model = Segmamba_hybrid_gsc_KAN_PE(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    # elif model_type == 'Segmamba_hybrid_gsc_rm_fkan':
    #     model = Segmamba_hybrid_gsc_rm_fkan(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    # elif model_type == 'Segmamba_hybrid_gsc_KAN_PE_rm_fkan':
    #     model = Segmamba_hybrid_gsc_KAN_PE_rm_fkan(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4

    # elif model_type == 'Segmamba_hybrid_gsc_KAN_PE_rm_fkan_ds':
    #     model = Segmamba_hybrid_gsc_KAN_PE_rm_fkan_ds(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4





    elif model_type == 'Segmamba_hybrid_gsc_KAN_PE_ds':
        model = Segmamba_hybrid_gsc_KAN_PE_ds(
            in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384], spatial_dims=3,)
        lr = 1e-4  
    
    # elif model_type == 'Segmamba_hybrid_gsc_KAN_PE_ds_SPATIAL':
    #     model = Segmamba_hybrid_gsc_KAN_PE_ds_SPATIAL(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4   

    # elif model_type == 'Segmamba_hybrid_gsc_MLP_PE_ds':
    #     model = Segmamba_hybrid_gsc_MLP_PE_ds(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4 



        
         

    # elif model_type == 'segmamba_hybrid_gsc_KAN_PE_EffKan':
    #     model = segmamba_hybrid_gsc_KAN_PE_EffKan(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4   

    # elif model_type == 'Segmamba_hybrid_gsc_KAN_PE_ds_flip':
    #     model = Segmamba_hybrid_gsc_KAN_PE_ds_flip(
    #         in_chans=config.n_channels, out_chans=config.n_labels, depths=[2, 2, 2, 2],
    #         feat_size=[48, 96, 192, 384], spatial_dims=3,)
    #     lr = 1e-4  

    # elif model_type == 'TransUNet_fJNB':
    #     model = TransUNet_KAN_fJNB(n_channels=config.n_channels, n_classes=config.n_labels)
    #     lr = 1e-4  

    # elif model_type == 'SegViT_fKAN':
    #     model = SegViT_fKAN(
    #         in_chans=config.n_channels,
    #         out_chans=config.n_labels,
    #         img_size=config.img_size,       
    #         vit_name="R50-ViT-B_16",        
    #     )
    #     lr = 1e-4

    else: 
        raise TypeError('Please enter a valid name for the model type')

    if model_type == 'SwinUnet':            
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    elif model_type == 'SMESwinUnet':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)  # Choose optimize


    model = model.cuda()
    print("Model Loaded!!")

    # from thop import profile

    # dummy_input = torch.randn(1, config.n_channels, config.img_size, config.img_size).cuda()
    # macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    # model_params = params / 1e6
    # model_gflops = macs / 1e9

    # print(f"Params: {model_params:.2f} M")
    # print(f"GFLOPs: {model_gflops:.2f} G")

    checkpoint_path = os.path.join(config.model_path, f'best_model-{model_type}.pth.tar')
    # checkpoint_path = os.path.join(config.model_path, f'best_model-{model_type}.pth.tar')
    print("Checkpoint path:", checkpoint_path)
    print("Exists:", os.path.isfile(checkpoint_path))

    start_epoch = 0
    max_dice = 0.0
    best_epoch = 1

    if resume and os.path.isfile(checkpoint_path):
        # max_dice = 0.9113
        logger.info(f" Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['epoch'] + 1
        # max_dice = checkpoint.get('val_loss', 0.0)
        max_dice = checkpoint.get('val_dice', 0.0)
        best_epoch = start_epoch

        logger.info(f" Model type: {model_type}")
        logger.info(f" Resuming from epoch: {checkpoint['epoch']}")
        logger.info(f" Last best dice score: {max_dice:.4f}")
        logger.info(f" Optimizer state and model weights loaded successfully")
        logger.info(f" Continuing training on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print("Resumed start_epoch:", start_epoch, "max_dice:", max_dice, "best_epoch:", best_epoch)




    logger.info('Training on ' +str(os.uname()[1]))
    logger.info('Training using GPU : '+torch.cuda.get_device_name(torch.cuda.current_device()))
    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])


    criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5, n_labels=config.n_labels)
    if model_type == 'Segmamba' or model_type == 'SegViT_fKAN':
        criterion = BinaryDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    # elif model_type == 'Segmamba_hybrid_gsc_ds' or model_type == 'Segmamba_hybrid_gsc_KAN_PE_rm_fkan_ds' or model_type == 'Segmamba_hybrid_gsc_KAN_PE_ds' or model_type == 'Segmamba_hybrid_gsc_KAN_PE_ds_flip':
    elif model_type in (
                            'Segmamba_hybrid_gsc_ds',
                            'Segmamba_hybrid_gsc_KAN_PE_rm_fkan_ds',
                            'Segmamba_hybrid_gsc_KAN_PE_ds',
                            'Segmamba_hybrid_gsc_KAN_PE_ds_flip',
                            'Segmamba_hybrid_gsc_MLP_PE_ds',
                            'Segmamba_hybrid_gsc_KAN_PE_ds_SPATIAL'
                        ):
                            
        # Deep supervision wrapper:
        # assume SegMamba returns: (main, ds1, ds2, ds3)
        base_loss = WeightedDiceBCE(
            dice_weight=0.5,
            BCE_weight=0.5,
            n_labels=config.n_labels
        )
        print("Using Deep Supervision Wrapper")
        criterion = DSAdapterLoss(
            base_loss=base_loss,
            ds_weights=(0.5, 0.3, 0.2),  # weights for ds1, ds2, ds3
            main_weight=1.0              # weight for main output
        )
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)
         
    # if model_type == 'UNeXt':
    #     # criterion = WeightedDiceBCE(dice_weight=1,BCE_weight=1, n_labels=config.n_labels)
    #     criterion = WeightedDiceBCE(dice_weight=1,BCE_weight=0.5, n_labels=config.n_labels)
    #     lr_scheduler = torch_lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)
    
    # elif model_type == 'UNext_CMRF_GS_Wavelet':
    #     criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5, n_labels=config.n_labels)
    #     lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)

     
    ### For Unext
    # criterion = WeightedDiceBCE(dice_weight=1,BCE_weight=0.5, n_labels=config.n_labels)

    # criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5, n_labels=config.n_labels)
    # criterion = WeightedDiceBCEHausdorff(dice_weight=0.4,BCE_weight=0.4,hausdorff_weight=0.2, n_labels=config.n_labels)

    # GAB Deep supervision wrapper
    # criterion = DSAdapterLoss(
    #     base_loss=WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5, n_labels=config.n_labels),
    #     ds_weights=(0.2, 0.3, 0.4, 0.5),   # match your preferred scheme
    #     main_weight=1.0
    # )

    # if config.cosineLR is True:
    #     lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
    #     if model_type == 'UNeXt':
    #         lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)

    if config.cosineLR is True:
        dummy = True
        # ACC-UNet and others
        # lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)
        # lr_scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
        # lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=0.00001)
        
        # UNext
        # lr_scheduler = torch_lr_scheduler.CosineAnnealingLR(optimizer, T_max=400, eta_min=1e-5)


    else:
        lr_scheduler =  None
    if tensorboard:
        log_dir = config.tensorboard_folder
        logger.info('log dir: '.format(log_dir))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    # max_dice = 0.0
    # best_epoch = 1
    print("Begin Training!!")
    # for epoch in range(config.epochs):  # loop over the dataset multiple times
    for epoch in range(start_epoch, config.epochs):  # resume if needed
        logger.info('\n========= Epoch [{}/{}] ========='.format(epoch + 1, config.epochs + 1))
        logger.info(config.session_name)
        # train for one epoch
        model.train(True)
        logger.info('Training with batch size : {}'.format(batch_size))
        train_one_epoch(train_loader, model, criterion, optimizer, writer, epoch, None, model_type, logger)
        # evaluate on validation set
        logger.info('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_dice = train_one_epoch(val_loader, model, criterion,
                                            optimizer, writer, epoch, lr_scheduler,model_type,logger)

        # =============================================================
        #       Save best model
        # =============================================================
        if val_dice > max_dice:
                #if epoch+1 > 5:
                logger.info('\t Saving best model, mean dice increased from: {:.4f} to {:.4f}'.format(max_dice,val_dice))
                # print(f"saving best model at path: {config.model_path}")
                max_dice = val_dice
                best_epoch = epoch + 1
                # save_checkpoint({'epoch': epoch,
                #                  'best_model': True,
                #                  'model': model_type,
                #                  'state_dict': model.state_dict(),
                #                  'val_loss': val_loss,
                #                  'optimizer': optimizer.state_dict()}, config.model_path)#+f'_{epoch}')
                save_checkpoint({
                                    'epoch': epoch,
                                    'best_model': True,
                                    'model': model_type,
                                    'state_dict': model.state_dict(),
                                    'val_loss': val_loss,
                                    'val_dice': val_dice,          # NEW
                                    'optimizer': optimizer.state_dict()
                                }, config.model_path)

                
                # model.save(config.model_path+f'/best_model-{model_type}.pth')
        else:
            logger.info('\t Mean dice:{:.4f} does not increase, '
                        'the best is still: {:.4f} in epoch {}'.format(val_dice,max_dice, best_epoch))
        early_stopping_count = epoch - best_epoch + 1
        logger.info('\t early_stopping_count: {}/{}'.format(early_stopping_count,config.early_stopping_patience))

        if early_stopping_count > config.early_stopping_patience:
            logger.info('\t early_stopping!')
            break

    return model


if __name__ == '__main__':
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    if not os.path.isdir(config.save_path):
        os.makedirs(config.save_path)

    print("Check 1")
    # if os.path.isfile(config.logger_path):
    #     import sys
    #     sys.exit()

    print("Sys Exit Bypass")
    logger = logger_config(log_path=config.logger_path)

    print("Logger Configured!!")
    # model = main_loop(model_type=config.model_name, tensorboard=True)
    model = main_loop(model_type=config.model_name, tensorboard=True, resume=True)

    
    fp = open('log.log','a')
    fp.write(f'{config.model_name} on {config.task_name} completed\n')
    fp.close()
    