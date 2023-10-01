import os
import shutil
import gc
import matplotlib.pyplot as plt
import random
import logging
logging.disable(logging.WARNING)
from itertools import chain
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR, CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.autograd import gradcheck
from torchvision import transforms, models
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
#import wandb

from utils import *
from running_wandb import yaml_load_with_wandb
from preprocess_nlp import Flicker8K, get_loaders
from model import Encoder, Decoder, EncoderDecoderWrapper




def get_scheduler(cfg, optimizer):
    if cfg.scheduler == 'one_cycle':
        scheduler = OneCycleLR(optimizer, max_lr= cfg.max_lr, steps_per_epoch=len(train_loader), epochs=cfg.epochs, verbose=True, total_steps = len(train_loader) *cfg.epochs)

    elif cfg.scheduler == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode=cfg.mode, factor=cfg.factor, patience=cfg.plateau_patience)
            
    elif cfg.scheduler=='cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr, last_epoch=-1)
        
    elif cfg.scheduler=='cosine_annealing_warm_restarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=1, eta_min=cfg.min_lr, last_epoch=-1)

    elif cfg.scheduler == 'cyclical_lr':
        cheduler = CyclicLR(optimizer,base_lr=cfg.min_lr,max_lr=cfg.max_lr, step_size_up=int(4 * (len(X_train_data) / cfg.batch_size)),
                                    cycle_momentum=False)
    else:
        raise NotImplementedError(cfg.scheduler)
    
    return scheduler


def get_optimizer(cfg):

    # to train encoder and decoder with same optimizer :  optimizer = optim.Adam(chain(encoder.parameters(), decoder.parameters()))

    if cfg.optim == 'SGD':
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=cfg.encoder_lr,
                                weight_decay=cfg.weight_decay, nesterov=True, momentum=cfg.momentum) if cfg.FINETUNE_ENCODER else None

        decoder_optimizer = optim.SGD(decoder.parameters(), lr=cfg.decoder_lr,
                                weight_decay=cfg.weight_decay, nesterov=True, momentum=cfg.momentum)
     
    elif cfg.optim  == 'Adam':
        # Using AdamW https://www.fast.ai/2018/07/02/adam-weight-decay/
        encoder_optimizer = optim.AdamW(encoder.parameters(), lr=cfg.encoder_lr, weight_decay=cfg.weight_decay) if cfg.FINETUNE_ENCODER else None
        decoder_optimizer = optim.AdamW(decoder.parameters(), lr=cfg.decoder_lr, weight_decay=cfg.weight_decay)

    else:
        raise NotImplementedError(cfg.optim)
    return encoder_optimizer, decoder_optimizer


# 2.90E-04  Best lr=  0.14283914439298 , Min lr=  0.014283914439298
# 1.10E-03  Best lr=  0.008497534359086439 , Min lr=  0.0008497534359086439 
# 3.85E-01  Best lr=  0.012618568830660204 , Min lr=  0.0012618568830660205
# 3.74E-02  Best lr=  0.2612675225563329 , Min lr=  0.02612675225563329

def find_lr(model, train_loader, device, criterion, optimizer, start_lr=1e-6, end_lr=0.5, num_iter=500):
    """ Uses Fastai tweaked version of Leslie Smith LR finder method.
    """
    
    lr_finder = LRFinderWrapper(model, optimizer, criterion, device=device)

    lr_finder.range_test(train_loader, start_lr=start_lr, end_lr=end_lr, num_iter=num_iter, step_mode="exp")

    lr_finder.plot(log_lr=False)[0].figure.savefig("lr_finder.png")

    best_lr =  lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
    return lr_finder.history, best_lr



if __name__ == '__main__':
 
    # get config yaml & setup wandb function
    cfg = yaml_load_with_wandb("config-defaults.yaml",use_wandb=False)

    # GPU check
    use_cuda = cfg.cuda and torch.cuda.is_available()
    if cfg.cuda and torch.cuda.is_available():
        print("Cuda enabled and available")
    elif cfg.cuda  and not torch.cuda.is_available():
        print("Cuda enabled but not available, CPU is used.")
    elif not cfg.cuda :
        print("Cuda disabled")
    device = torch.device("cuda" if use_cuda else "cpu")

    # Seed everything
    seed_all(cfg.seed)


    # Data loaders
    train_loader, val_loader, test_loader, processor = get_loaders(cfg, cuda=use_cuda)


    # Setup models
    encoder = Encoder(cfg, device).to(device)
    decoder = Decoder(cfg, processor.vocab, device).to(device)


    # Get criterion
    criterion = nn.CrossEntropyLoss(ignore_index=processor.vocab[cfg.PAD_TOKEN]).to(device)


    # Setup opimizer: using same optimizer and lr scheduler types for the encoder and decoder
    encoder_optimizer, decoder_optimizer = get_optimizer(cfg)


    # Setup LR Finder
    if cfg.lr_finder:
        from torch_lr_finder import LRFinder
        print('Running LR finder ... ')
        # LR finder for encoder-decoder model while freezing encoder CNN weights 
        model_wrapper = EncoderDecoderWrapper(encoder, decoder).to(device)
        optimizer = torch.optim.AdamW(model_wrapper.parameters(), lr=0.01, weight_decay=cfg.weight_decay)
        custom_train_iter = CustomTrainIter(train_loader)
        history, best_lr = find_lr(model_wrapper, custom_train_iter, device, criterion, optimizer)
        min_lr = best_lr/10
        print("Best lr= ", best_lr, ", Min lr= ", min_lr)


    # Setup scheduler
    decoder_scheduler = get_scheduler(cfg, decoder_optimizer) if cfg.use_scheduler else None
    if encoder_optimizer is None or not cfg.use_scheduler:
        encoder_scheduler = None
    else:
        encoder_scheduler = get_scheduler(cfg, encoder_optimizer) 


    # Setup mixed precision training
    if cfg.use_amp:
        from apex import amp
        [encoder, decoder], [encoder_optimizer, decoder_optimizer] = amp.initialize([encoder, decoder], [encoder_optimizer, decoder_optimizer], opt_level="O1", verbosity=0)
        print(('Using fp16 training ...'))
        

    # Setup Early Stopping    
    early_stop = EarlyStopping(patience=cfg.patience, mode='max') if cfg.early_stop else None
    

    # load checkpoint to resume training 
    if cfg.resume_training:
        encoder, decoder, encoder_optimizer, decoder_optimizer, start_epoch  = load_checkpoint(cfg, encoder, decoder, enc_optimizer=encoder_optimizer, dec_optimizer=decoder_optimizer, resume_training=True)
        
    elif not cfg.resume_training:
        start_epoch = 1 
    
    
    # run training 
    if cfg.train_network or cfg.resume_training:
        print(f'Training for {cfg.epochs} epochs ...')
        
        wandb.watch(encoder, log="all", log_freq=cfg.log_wandb_freq)
        wandb.watch(decoder, log="all", log_freq=cfg.log_wandb_freq)
        train_losses, test_losses = train_all(cfg, encoder, decoder, device, train_loader, val_loader, encoder_optimizer, decoder_optimizer, criterion, start_epoch, processor, cfg.track_metric, encoder_scheduler=encoder_scheduler, decoder_scheduler=decoder_scheduler, early_stop=early_stop)
        plot_losses(train_losses, test_losses, cfg.plot_name)

    
    # run inference
    elif cfg.inference :
        print('Running inference')
        encoder, decoder, enc_optimizer, dec_optimizer, epoch_num = load_checkpoint(cfg, encoder, decoder)

        caption = inference_beam_search(encoder, decoder, processor.idx2token, processor.vocab, cfg, device, image_path=cfg.inference_image) #data_loader=test_loader 
        print('Caption is: ', caption)


