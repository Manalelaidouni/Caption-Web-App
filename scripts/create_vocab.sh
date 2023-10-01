#!/bin/bash

echo "Activating conda environment ... "

cd C:/ProgramData/miniconda3/etc/profile.d
source \conda.sh
conda activate base

cd C:/Users/Administrator/Desktop/MyFinalCaptionPipeline

python - <<EOF
import pickle
from running_wandb import yaml_load_with_wandb  
from preprocess_nlp import Preprocessor

cfg = yaml_load_with_wandb("config_defaults.yaml", use_wandb=False)
processor = Preprocessor(cfg)

with open('shell_vocab.pickle', 'wb') as f:
    pickle.dump(processor.vocab, f, pickle.HIGHEST_PROTOCOL)

with open('shell_idx2token.pickle', 'wb') as f:
    pickle.dump(processor.idx2token, f, pickle.HIGHEST_PROTOCOL)

EOF

echo "Done creating vocab.pickle and idx2token.pickle."