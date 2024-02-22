import argparse
import wandb
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from strategies.architecture.wandb.hawkes_training_wandb import train as hawkes_train
from strategies.architecture.wandb.flows_training_wandb import train as flows_train
from strategies.architecture.wandb.gmm_training_wandb import train as gmm_train

import myTransformer.Utils as Utils
from data_preparation import initial_dataloader_preparation
from myTransformer.Models import Transformer, TransformerMixure


MODEL_PATH_PREFIX = "trainedModels\\tf"
MODEL_PATH_SUFFIX = ".pth.tar"

SWEEP_NAME_GMM = "gmm_depend"
SWEEP_NAME_HAWKES = "hawkes_depend"
SWEEP_NAME_FLOWS = "flows_test_noHawk"

#python MainWAndB.py -architecture

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-architecture', type=str, default= 'hawkes')
    opt = parser.parse_args()
    set_seed(42)
    init_dict = {
        "gmm": init_gmm,
        "hawkes": init_hawkes,
        "flows": init_flows
    }
    wandb.login()
    sweep_id = init_dict.get(opt.architecture, "hawkes")()

    train_dict = {
        "gmm": gmm_train,
        "hawkes": hawkes_train,
        "flows": flows_train
    }
    train = train_dict.get(opt.architecture, "hawkes")
    wandb.agent(sweep_id, train, count=200)

def init_gmm():
    print()
    print("Initialize gmm transformer...")
    print()
    sweep_config = {
    'method': 'grid'
    }
    metric = {
    'name': 'test_loss',
    'goal': 'minimize'   
    }

    parameters_dict = {
        'learning_rate': {
            'values': [0.0001]
        },
        'epochs': {
            'value': 50
        },
        'num_mix_components': {
            'values': [4,5,6] #d_inner = 4*d_model
        },
        'n_layers': {
            'values': [3,4,5,6]
        },
        'batch_size': {
            'value': 4
        },
        'n_head': {
            'values': [2,4,8]
        },
        'd_k': {
            'values': [32, 64] # also used for d_v
        },
        'dropout': {
            'value': 0.1
        },
        'smooth': {
            'value': 0
        },
        'device':{
            'value': 'cuda'
        },
        'data': {
            'value': 'data/Breakfast/'
        }
    }
    
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=SWEEP_NAME_GMM)
    return sweep_id

def init_hawkes():
    print()
    print("Initialize hawkes transformer...")
    print()
    sweep_config = {
    'method': 'grid'
    }
    metric = {
    'name': 'test_loss',
    'goal': 'minimize'   
    }
    parameters_dict = {
        'learning_rate': {
            'value': 0.0001
        },
        'epochs': {
            'value': 50
        },
        'n_layers': {
            'values': [3]
        },
        'batch_size': {
            'value': 4
        },
        'd_rnn': {
            'values': [0, 128]
        },
        'n_head': {
            'values': [8]
        },
        'd_k': {
            'values': [128] # also used for d_v
        },
        'dropout': {
            'value': 0.1
        },
        'smooth': {
            'value': 0
        },
        'activation': {
            'values': ['softplus']
        },
        'device':{
            'value': 'cuda'
        },
        'data': {
            'value': 'data/Breakfast/'
        }
    }
    
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=SWEEP_NAME_HAWKES)
    return sweep_id

def init_flows():
    print("Initialize flow transformer...")
    sweep_config = {
    'method': 'grid'
    }
    metric = {
    'name': 'test_loss',
    'goal': 'minimize'   
    }
    parameters_dict = {
        'learning_rate': {
            'value': 0.0001
        },
        'epochs': {
            'value': 50
        },
        'n_layers': {
            'values': [3,4,5,6]
        },
        'batch_size': {
            'value': 4
        },
        'd_rnn': {
            'values': [0, 128]
        },
        'n_head': {
            'values': [2,4,8]
        },
        'd_k': {
            'values': [32, 64, 128, 256] # also used for d_v
        },
        'dropout': {
            'value': 0.1
        },
        'smooth': {
            'value': 0
        },
        'activation': {
            'values': ['softplus']
        },
        'device':{
            'value': 'cuda'
        },
        'data': {
            'value': 'data/Breakfast/'
        }
    }
    
    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project=SWEEP_NAME_FLOWS)
    return sweep_id

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f'Set SEED to {SEED}')

if __name__ == '__main__':
    main()