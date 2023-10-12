#!/bin/bash

conda_path="$1"
conda_env_name="$2"
project_path="$PWD"

echo  "Activating conda $conda_env_name environment ... "
cd $conda_path
source \conda.sh
conda activate $conda_env_name
cd $project_path && cd ..

pip install git+https://github.com/davidtvs/pytorch-lr-finder.git

python - <<EOF
import pickle
import torch
import torch.nn as nn
from torch_lr_finder import LRFinder, TrainDataLoaderIter

from model import Encoder, Encoder, Decoder, EncoderDecoderWrapper
from running_wandb import yaml_load_with_wandb   
from utils import LRFinderWrapper, CustomTrainIter
from train import find_lr
from preprocess_nlp import  get_loaders

cfg = yaml_load_with_wandb("config_defaults.yaml", use_wandb=False)
device = torch.device("cpu")

train_loader, _, _, processor = get_loaders(cfg, cuda=False)
encoder = Encoder(cfg, device).to(device)
decoder = Decoder(cfg, processor.vocab, device).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=processor.vocab[cfg.PAD_TOKEN]).to(device)

print('Creating LR finder ... ')

model_wrapper = EncoderDecoderWrapper(encoder, decoder).to(device)
optimizer = torch.optim.AdamW(model_wrapper.parameters(), lr=0.01, weight_decay=cfg.weight_decay)
custom_train_iter = CustomTrainIter(train_loader)

print('Running LR finder ... ')
history, best_lr = find_lr(model_wrapper, custom_train_iter, device, criterion, optimizer)
min_lr = best_lr/10
print("Best lr= ", best_lr, ", Min lr= ", min_lr)

EOF


