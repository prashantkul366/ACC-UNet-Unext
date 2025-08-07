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
from Load_Dataset import RandomGenerator,ValGenerator,ImageToImage2D
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nets.ACC_UNet import ACC_UNet
from nets.MResUNet1 import MultiResUnet
from nets.SwinUnet import SwinUnet
from nets.UNet_base import UNet_base
from nets.SMESwinUnet import SMESwinUnet
from nets.UCTransNet import UCTransNet


##################### NEW ARCHS ######################

from nets.UNext import UNext
from nets.archs.archs_InceptionNext_MLFC import UNext_InceptionNext_MLFC
from nets.archs.UNext_CMRF import UNext_CMRF
from nets.archs.UNext_CMRF_enc_dec import UNext_CMRF_enc_dec
from nets.archs.UNext_CMRF_enc_MLFC import UNext_CMRF_enc_MLFC
######################################################


from torch.utils.data import DataLoader
import logging
import json
from Train_one_epoch import train_one_epoch
import Config as config
from torchvision import transforms
from utils import CosineAnnealingWarmRestarts, WeightedDiceBCE

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
        model = UNext_InceptionNext_MLFC(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4  

    elif model_type == 'UNext_CMRF':
        model = UNext_CMRF(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4

    elif model_type == 'UNext_CMRF_enc_dec':
        model = UNext_CMRF_enc_dec(n_channels=config.n_channels, n_classes=config.n_labels)
        # lr = 1e-4

    elif model_type == 'UNext_CMRF_enc_MLFC':
        model = UNext_CMRF_enc_MLFC(n_channels=config.n_channels, n_classes=config.n_labels)

        
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

    checkpoint_path = os.path.join(config.model_path, f'best_model-{model_type}.pth.tar')
    start_epoch = 0
    max_dice = 0.0
    best_epoch = 1

    if resume and os.path.isfile(checkpoint_path):
        logger.info(f" Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        max_dice = checkpoint.get('val_loss', 0.0)
        best_epoch = start_epoch

        logger.info(f" Model type: {model_type}")
        logger.info(f" Resuming from epoch: {checkpoint['epoch']}")
        logger.info(f" Last best dice score: {max_dice:.4f}")
        logger.info(f" Optimizer state and model weights loaded successfully")
        logger.info(f" Continuing training on GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")




    logger.info('Training on ' +str(os.uname()[1]))
    logger.info('Training using GPU : '+torch.cuda.get_device_name(torch.cuda.current_device()))
    # if torch.cuda.device_count() > 1:
    #     print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
    # model = nn.DataParallel(model, device_ids=[0])
    criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5, n_labels=config.n_labels)
    
    if config.cosineLR is True:
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4)
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
                max_dice = val_dice
                best_epoch = epoch + 1
                save_checkpoint({'epoch': epoch,
                                 'best_model': True,
                                 'model': model_type,
                                 'state_dict': model.state_dict(),
                                 'val_loss': val_loss,
                                 'optimizer': optimizer.state_dict()}, config.model_path)#+f'_{epoch}')
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
    if os.path.isfile(config.logger_path):
        import sys
        sys.exit()

    print("Sys Exit Bypass")
    logger = logger_config(log_path=config.logger_path)

    print("Logger Configured!!")
    # model = main_loop(model_type=config.model_name, tensorboard=True)
    model = main_loop(model_type=config.model_name, tensorboard=True, resume=False)

    
    fp = open('log.log','a')
    fp.write(f'{config.model_name} on {config.task_name} completed\n')
    fp.close()
    