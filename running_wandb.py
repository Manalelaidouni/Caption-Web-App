import os
import yaml
from easydict import EasyDict as edict
from dotmap import DotMap
#from utils import parse_arguments

import argparse

# folder to load config file
CONFIG_PATH = "./configurations"


def yaml_load_with_wandb(config_name,  use_wandb=True, config_path=None, parse_args=True):
    # change path to call config file from app.py
    config_path = config_path if config_path is not None else CONFIG_PATH

    if use_wandb:
        import wandb
        with open(os.path.join(config_path, config_name)) as yamlfile:
            # returns dict object
            confg = yaml.load(yamlfile, Loader=yaml.Loader)

        # setup wandb
        wandb.init(project=confg['project']['value'], tags=confg['tag']['value'], notes=confg['notes']['value'], allow_val_change=True)
        cfg = dict(wandb.config)
        cfg = DotMap(cfg)

    else:
        # access Yaml dictionary values as attributes
        # to not change code when not using wandb
        with open(os.path.join(config_path, 'config_defaults.yaml')) as yamlfile: 
            cfg = edict(yaml.load(yamlfile, Loader=yaml.Loader))

    
    # args passed last in CLI overrides ones in YAML config file
    if parse_args:
        args = parse_arguments()
        cfg = dict(cfg)
        # convert namespace object to dictionary
        args = vars(args) 
        cfg.update(args)
        cfg = edict(cfg)
   
    return cfg



def parse_arguments():
    parser = argparse.ArgumentParser(description='Command line interface for Caption web app.')

    parser.add_argument('--use_wandb', type=bool, default=False, required=False, help='Use Weights and Biases')
    parser.add_argument('--project', type=str, default='ImgCaption', required=False, help='Wandb project name')
    parser.add_argument('--tag', type=str, default='link to wandb', required=False, help='Wandb tag')
    parser.add_argument('--notes', type=str, default='training one cycle', required=False, help='Notes for the run')
    parser.add_argument('--log_wandb_freq', type=int, default=10, required=False, help='Frequency to log to Wandb')
    parser.add_argument('--parse_args', type=bool, default=True, required=False, help='If args should be parsed for a bash script')
    parser.add_argument('--checkpoint_fname', type=str, default='checkpoint_at_epoch_6', required=False, help='Checkpoint filename')
    parser.add_argument('--CHECKPOINT_PATH', type=str, default='.\checkpoint', required=False, help='Path to weights checkpoint')
    parser.add_argument('--IMG_PATH', type=str, default='.\Data\Flicker8k_Dataset', required=False, help='Path to image data')
    parser.add_argument('--TEXT_PATH', type=str, default='.\Data\Flicker8k_text', required=False, help='Path to text data')
    parser.add_argument('--GLOVE_PATH', type=str, default='.\Data\glove.6B.50d.txt', required=False, help='Path to GloVe embeddings')
    parser.add_argument('--use_scheduler', type=bool, default=False, required=False, help='Use learning rate scheduler')
    parser.add_argument('--scheduler', type=str, default='one_cycle', required=False, help='Learning rate scheduler type')
    parser.add_argument('--sched_patience', type=int, default=3, required=False, help='Scheduler patience')
    parser.add_argument('--lr_finder', type=bool, default=False, required=False, help='Use learning rate finder')
    parser.add_argument('--weight_decay', type=float, default=0.01, required=False, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, required=False, help='Dropout rate')
    parser.add_argument('--checkpoint', type=bool, default=True, required=False, help='Checkpoint saving')
    parser.add_argument('--save_to_wandb', type=bool, default=True, required=False, help='Save to Weights and Biases')
    parser.add_argument('--seed', type=int, default=1, required=False, help='Random seed')
    parser.add_argument('--clip', type=float, default=-1, required=False, help='Gradient clipping threshold')
    parser.add_argument('--cuda', type=bool, default=True, required=False, help='Use CUDA if available')
    parser.add_argument('--early_stop', type=bool, default=True, required=False, help='Early stopping')
    parser.add_argument('--patience', type=int, default=10, required=False, help='Patience for early stopping')
    parser.add_argument('--mode', type=str, default='min', required=False, help='Mode for early stopping (min or max)')
    parser.add_argument('--validate', type=bool, default=False, required=False, help='Validation mode')
    parser.add_argument('--inference', type=bool, default=False, required=False, help='Inference mode')
    parser.add_argument('--train_network', type=bool, default=False, required=False, help='Training mode')
    parser.add_argument('--num_workers', type=int, default=1, required=False, help='Number of workers for data loading')
    parser.add_argument('--inference_image', type=str, default=None, required=False, help='Path to image to run inference on')
    parser.add_argument('--learning_rate', type=float, default=0.01, required=False, help='Learning rate')

    parser.add_argument('--torch_hub_dir', type=str, default='pytorch/vision:v0.8.0', help='Torch Hub directory')
    parser.add_argument('--torch_hub_model', type=str, default='resnet50', help='Torch Hub model')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model')
    parser.add_argument('--FINETUNE_ENCODER', type=bool, default=False, help='Finetune encoder')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Encoder dimension')
    parser.add_argument('--decoder_dim', type=int, default=512, help='Decoder dimension')
    parser.add_argument('--attention_dim', type=int, default=512, help='Attention dimension')
    parser.add_argument('--embed_dim', type=int, default=50, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--depth', type=int, default=2048, help='Depth')
    parser.add_argument('--encoder_size', type=int, default=18, help='Encoder size')
    parser.add_argument('--mixed_training', type=bool, default=False, help='Mixed training')
    parser.add_argument('--batch_first', type=bool, default=True, help='Batch first')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('--bidirectional', type=bool, default=False, help='Bidirectional LSTM')
    parser.add_argument('--use_glove_embeddings', type=bool, default=True, help='Use GloVe embeddings')
    parser.add_argument('--finetune_embedding', type=bool, default=False, help='Finetune embeddings')
    parser.add_argument('--use_attention', type=bool, default=False, help='Use attention')
    parser.add_argument('--use_one_network', type=bool, default=False, help='Use one network')
    parser.add_argument('--teacher_forcer', type=float, default=0.5, help='Teacher forcing ratio')
    parser.add_argument('--encoder_lr', type=float, default=0.05, help='Encoder learning rate')
    parser.add_argument('--decoder_lr', type=float, default=0.05, help='Decoder learning rate')
    parser.add_argument('--remove_punct', type=bool, default=True, help='Remove punctuation')
    parser.add_argument('--lemmatize', type=bool, default=True, help='Lemmatize')
    parser.add_argument('--stemmize', type=bool, default=False, help='Stemmize')
    parser.add_argument('--remove_stopwords', type=bool, default=False, help='Remove stopwords')
    parser.add_argument('--remove_numbers', type=bool, default=True, help='Remove numbers')
    parser.add_argument('--track_metric', type=str, default='bleu', help='Tracking metric')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle data')
    parser.add_argument('--transform_input', type=bool, default=True, help='Transform input data')
    parser.add_argument('--plot_name', type=str, default='train vs validation', help='Plot name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--use_amp', type=bool, default=False, help='Use Automatic Mixed Precision (AMP)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer')
    parser.add_argument('--resume_training', type=bool, default=False, help='Resume training')

    args = parser.parse_args()
    return args
