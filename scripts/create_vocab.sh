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
from running_wandb import yaml_load_with_wandb  
from preprocess_nlp import Preprocessor

cfg = yaml_load_with_wandb("config_defaults.yaml", use_wandb=False)
processor = Preprocessor(cfg)

with open('vocab.pickle', 'wb') as f:
    pickle.dump(processor.vocab, f, pickle.HIGHEST_PROTOCOL)

with open('idx2token.pickle', 'wb') as f:
    pickle.dump(processor.idx2token, f, pickle.HIGHEST_PROTOCOL)

EOF

echo "Done creating vocab.pickle and idx2token.pickle."