import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from enum import Enum
from strategies.architecture.flow_training import FlowTraining
from strategies.architecture.gmm_training import GMM_Training
from strategies.architecture.hawkes_training import HawkesTraining

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils
from data_preparation import initial_dataloader_preparation
from myTransformer.Models import Transformer, TransformerMixure

from strategies.evaluation.longterm import Longterm_Strategy
from strategies.evaluation.longterm_gmm import Longterm_GMM_Strategy
from strategies.evaluation.shortterm_flows import Shortterm_Flows_Strategy
from strategies.evaluation.shortterm_hawkes import Shortterm_Hawkes_Strategy

MODEL_PATH_PREFIX = "trainedModels\\tf"
MODEL_PATH_SUFFIX = ".pth.tar"
LONGEST_TEST_ACTION_SEQUENCE = 20
ALPHA = .3
LIST_OF_BETA_VALUES = [0.1, 0.2, 0.3, 0.5]
BREAKFAST_FRAME_RATE = 15

# proactive (https://github.com/data-iitd/proactive/): 
# python MyMain.py -data data/Breakfast/ -batch 4 -n_head 4 -n_layers 4 -d_model 64 -d_inner 256 -epoch 50 -rnn_layer False -timeframe st -activation default -architecture flows

# transformer hawkes (https://github.com/SimiaoZuo/Transformer-Hawkes-Process/tree/master):
# python MyMain.py -data data/Breakfast/ -batch 4 -n_head 4 -n_layers 4 -d_model 64 -d_inner 256 -epoch 50 -rnn_layer True -timeframe st -activation softplus -architecture hawkes
# python MyMain.py -data data/Breakfast/ -batch 4 -n_head 4 -n_layers 4 -d_model 512 -d_rnn 64 -d_inner 1024 -d_k 512 -d_v 512 -dropout 0.1 -lr 1e-4 -smooth 0.1 -epoch 100 -rnn_layer True -timeframe st -activation softplus -architecture hawkes

# longterm evaluation on proactive
# python MyMain.py -data data/Breakfast/ -batch 4 -n_head 4 -n_layers 4 -d_model 64 -d_inner 256 -epoch 50 -rnn_layer False -timeframe lt -activation softplus -architecture flows

# longterm evaluation on transformer hawkes
# python MyMain.py -data data/Breakfast/ -batch 4 -n_head 4 -n_layers 4 -d_model 64 -d_inner 256 -epoch 50 -rnn_layer True -timeframe lt -activation softplus -architecture hawkes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', default = "data/Breakfast/")
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=32)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    ### add argument for switching between evaluations
    parser.add_argument('-rnn_layer', type=boolean_string, default= 'False')
    parser.add_argument('-mix', type=int, default= 16)

    parser.add_argument('-timeframe', type=str, default= 'lt')
    parser.add_argument('-activation', type=activation_string, default= 'default')

    parser.add_argument('-architecture', type=architecture_string, default= 'gmm')

    opt = parser.parse_args()

    opt.device = torch.device('cuda')

    set_seed(42)
    
    architecture = opt.architecture
    print(architecture)
    activation = opt.activation
    rnn_flag = opt.rnn_layer
    rnn_descriptor = "rnnOff"

    if(rnn_flag):
        rnn_descriptor = "rnnOn"
    
    # pick correct string for timeframe so that the evaluation strategy is correct
    timeframe = opt.timeframe
    if(opt.timeframe == 'st'):
        if(architecture == Architecture.HAWKES):
            timeframe += "_h"
        else:
            timeframe += "_f"
    print(timeframe)
    timeframe = timeframe_string(timeframe)
    print(timeframe)
    evaluation_strategy = _set_eval_strategy(timeframe)
    train = _set_training_strategy(architecture)

    if(architecture == Architecture.GMM):
        model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal = prepare_gmmModel(opt)
        if(timeframe == Timeframe.LONGTERM):
            evaluation_strategy = _set_eval_strategy(Timeframe.LONGTERM_GMM)
    else:
        model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal = prepare_proactive(opt)

    config = architecture.value + "_" + activation + "_" +  rnn_descriptor
    model_path = MODEL_PATH_PREFIX + "_" + config + MODEL_PATH_SUFFIX

    # check for an existing trained model
    if Path(model_path).is_file():
        print("Trained transformer found. Initializing model parameters...")
        save_state = torch.load(model_path)
        model_state = save_state['model_state_dict']
        optimizer_state = save_state['optimizer_state_dict']
    else:
        print("No model found. Initiating training...")
        # a model is trained and saved to evaluate at a later time
        model_state, optimizer_state = train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)
        torch.save({
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            }, model_path)
        
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
          
    evaluation_strategy(model, optimizer, scheduler, opt)

def prepare_proactive(opt):
    trainloader, testloader, num_types, num_goals = initial_dataloader_preparation(opt)
 
    hawkes = False
    if(opt.architecture == Architecture.HAWKES):
        hawkes = True
    model = Transformer(
        num_types=num_types,
        num_goals=num_goals,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        activation=opt.activation,
        hawkes=hawkes
    )
    model.to(opt.device)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
        pred_loss_goal = Utils.LabelSmoothingLoss(opt.smooth, num_goals, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        pred_loss_goal = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    return model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal

def prepare_gmmModel(opt):
    trainloader, testloader, num_types, num_goals = initial_dataloader_preparation(opt)

    regularization = 1e-5  # L2 regularization parameter
    learning_rate = 1e-3   # Learning rate for Adam optimizer   

    model = TransformerMixure(
        num_types=num_types,
        num_goals=num_goals,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
        num_mix_components= opt.mix
    )
    model.to(opt.device)
    #optimizer from "intensity-free"
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)
    #part of proactive
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
        pred_loss_goal = Utils.LabelSmoothingLoss(opt.smooth, num_goals, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        pred_loss_goal = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    return model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal

def _set_eval_strategy(strat):
    if(strat == Timeframe.LONGTERM):
        lt = Longterm_Strategy()
        return lt.evaluate
    elif(strat == Timeframe.LONGTERM_GMM):
        lt_gmm = Longterm_GMM_Strategy()
        return lt_gmm.evaluate
    elif(strat == Timeframe.SHORTTERM_FLOWS):
        st_f = Shortterm_Flows_Strategy()
        return st_f.evaluate
    elif(strat == Timeframe.SHORTTERM_HAWKES):
        st_h= Shortterm_Hawkes_Strategy()
        return st_h.evaluate
    else:
        print("No evaluation specification found. Evaluating short term action prediciton.")
        st_f = Shortterm_Flows_Strategy()
        return st_f.evaluate

def timeframe_string(s):
    timeframe_map = {
        "st_f": Timeframe.SHORTTERM_FLOWS,
        "st_h": Timeframe.SHORTTERM_HAWKES,
        "lt": Timeframe.LONGTERM,
    }
    return timeframe_map.get(s, Timeframe.SHORTTERM_FLOWS)

class Timeframe(Enum):
    SHORTTERM_FLOWS = "st_f"
    SHORTTERM_HAWKES = "st_h"
    LONGTERM = "lt"
    LONGTERM_GMM = "lt_gmm"

def activation_string(s):
    activations = ["relu", "softplus", "elu", "default"]
    if(s in activations):
        return s
    print("No activation specification for time-predictor found. Defaulting to no activation.")
    return "default"

def _set_training_strategy(strat):
    if(strat == Architecture.FLOWS):
        ft = FlowTraining()
        return ft.train
    elif(strat == Architecture.HAWKES):
        ht = HawkesTraining()
        return ht.train
    elif(strat == Architecture.GMM):
        gmmt = GMM_Training()
        return gmmt.train
    else:
        print("No training specification found. Training with flows module.")
        ft = FlowTraining()
        return ft.train
    
def architecture_string(s):
    print(s)
    architecture_map = {
        "flows": Architecture.FLOWS,
        "hawkes": Architecture.HAWKES,
        "gmm": Architecture.GMM
    }
    return architecture_map.get(s, Architecture.FLOWS)

class Architecture(Enum):
    FLOWS = "flows"
    HAWKES = "hawkes"
    GMM = "gmm"

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

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

