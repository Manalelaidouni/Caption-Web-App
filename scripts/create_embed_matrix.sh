#!/bin/bash

echo "Activating conda environment ... "

cd C:/ProgramData/miniconda3/etc/profile.d
source \conda.sh
conda activate base

cd C:/Users/Administrator/Desktop/MyFinalCaptionPipeline

python - <<EOF

import pickle
import torch

from running_wandb import yaml_load_with_wandb   
from model import Decoder

cfg = yaml_load_with_wandb("config_defaults.yaml", use_wandb=False)

print('Loading processing tools ...')
vb = open("vocab.pickle",  'rb')
idx = open("idx2token.pickle",  'rb')
vocab = pickle.load(vb)
idx2token = pickle.load(idx)

device = torch.device("cpu")
decoder = Decoder(cfg, vocab, device).to(device)
print('Loading and processing Glove embeddings  ...')
embedding_matrix= decoder._get_glove_matrix(cfg, vocab)

print('Saving created embedding_matrix...')
torch.save(embedding_matrix, 'shell_embedding_matrix.pt')


EOF