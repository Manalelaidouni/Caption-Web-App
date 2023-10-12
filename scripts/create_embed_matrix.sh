#!/bin/bash

conda_path="$1"
conda_env_name="$2"
project_path="$PWD"

echo  "Activating conda $conda_env_name environment ... "
cd $conda_path
source \conda.sh
conda activate $conda_env_name
cd $project_path && cd ..

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