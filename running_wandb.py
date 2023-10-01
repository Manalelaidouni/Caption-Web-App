import os
import yaml
from easydict import EasyDict as edict
from dotmap import DotMap
from utils import parse_arguments

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
            print('No wandb, ', cfg, '\n\n')

    
    # args passed last in CLI overrides ones in YAML config file
    if parse_args:
        args = parse_arguments()
        cfg = dict(cfg)
        # convert namespace object to dictionary
        args =vars(args) 
        print('pre args: ', args, '\n\n')
        cfg.update(args)
        cfg=edict(cfg)
        print('post args: ', cfg, '\n\n')
   
    return cfg


if __name__ == '__main__' :
    cfg = yaml_load_with_wandb("config_defaults.yaml")
    print(cfg.weight_decay)
    print(cfg.epochs)

    from torchvision import datasets, transforms
    import torch

    transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    test_dataset = datasets.MNIST('../data', train=False,
                       transform=transform)

    # setup kwargs
    train_kwargs = {'batch_size': cfg.batch_size, 'shuffle': True, 
                                            'drop_last': True}
    test_kwargs = {'batch_size': cfg.test_batch_size,
                                        'drop_last' : True, 'shuffle': False}

    cuda = True
    if cuda:
        cuda_kwargs = {'num_workers': cfg.num_workers,
                'pin_memory': True,
                }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    iterator = iter(train_loader)
    x, y = next(iterator)
    print(x.shape, y.shape)