import pickle

from running_wandb import yaml_load_with_wandb
from preprocess_nlp import Preprocessor


cfg = yaml_load_with_wandb("config_defaults.yaml", use_wandb=False)
processor = Preprocessor(cfg)

with open('vocab.pickle', 'wb') as f:
    pickle.dump(processor.vocab, f, pickle.HIGHEST_PROTOCOL)

with open('idx2token.pickle', 'wb') as f:
    pickle.dump(processor.idx2token, f, pickle.HIGHEST_PROTOCOL)


