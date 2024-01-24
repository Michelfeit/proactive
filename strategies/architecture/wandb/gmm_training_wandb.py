from myTransformer.Models import TransformerMixure

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pprint
import wandb
from data_preparation import initial_dataloader_preparation
from tqdm import tqdm
from data_preparation import get_prediction_loader, load_prediciton_data, load_test_eos_data

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils
import myTransformer.Mixture_Utils as Mix_Utils

from strategies.architecture.training_strategy import Training_Strategy
from trim_process import EventData_Trim

TYPE_LOSS_SCALE = 1

def train(config=None):
    with wandb.init(config=config):
        print("Training hawkes transformer...")
        config = wandb.config
        pprint.pprint(config)
        trainloader, testloader, num_types, num_goals = initial_dataloader_preparation(config)

        regularization = 1e-5  # L2 regularization parameter
        learning_rate = 1e-3   # Learning rate for Adam optimizer   

        model = TransformerMixure(
            num_types=num_types,
            num_goals=num_goals,
            d_model=config.d_model,
            d_inner=config.d_model * 4,
            n_layers=config.n_layers,
            n_head=config.n_head,
            d_k=config.d_k,
            d_v=config.d_k,
            dropout=config.dropout,
            num_mix_components= config.num_mix_components
        )
        model.to(config.device)
        #optimizer from "intensity-free"
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate)
        #part of proactive
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

        if config.smooth > 0:
            pred_loss_func = Utils.LabelSmoothingLoss(config.smooth, num_types, ignore_index=-1)
            pred_loss_goal = Utils.LabelSmoothingLoss(config.smooth, num_goals, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            pred_loss_goal = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        test_acc_list = []
        test_goal_list = []
        test_mae_list = []

        best_loss = 0

        event_data = get_event_data(config)

        for epoch_i in range(config.epochs):
            print('[ Epoch', epoch_i, ']')
            time_loss, type_loss = train_epoch(model, trainloader, optimizer, pred_loss_func, pred_loss_goal, config, event_data)
            current_train_loss = time_loss + type_loss
            print("Train loss:", current_train_loss.item() , "with time_loss:", time_loss.item(), "and type_loss", type_loss.item())
            wandb.log({'time_loss_train': time_loss, 'type_loss_train': type_loss, 'train_loss': current_train_loss})

            time_loss, type_loss = eval_epoch(model, testloader, pred_loss_func, pred_loss_goal, config, event_data)
            current_loss = time_loss + type_loss
            print("Test loss:", current_loss.item() , "with time_loss:", time_loss.item(), "and type_loss", type_loss.item())
            wandb.log({'time_loss_test': time_loss, 'type_loss_test': type_loss})
            wandb.log({'test_loss': current_loss})

            if current_loss < best_loss:
                print("New best found!")
                best_loss = current_loss
                model_state = model.state_dict()
                optim_state = optimizer.state_dict()
            scheduler.step()
        print("Best loss is:", best_loss) 
    return model_state, optim_state

def train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt, event_data):
    """ Epoch operation in training phase. """
    model.train()
    
    total_time_loss = 0.0
    total_pred_loss = 0

    total_count = 0
    total_num_pred = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        optimizer.zero_grad()

        event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)

        enc_out, (prediction, mixture_enc) = model(event_type, event_time)

        pred_loss, pred_num_event = Utils.type_loss(prediction, event_type, pred_loss_func)
        #print("PRED:",pred_loss)
        gmm = Mix_Utils.get_inter_time_dist(model, opt, event_data, mixture_enc)
        time_loss = -Mix_Utils.log_probability(gmm, time_gap).mean()

        loss = time_loss + pred_loss
        #wandb.log({'batch_loss_train': loss})
        loss.backward()

        # training loop as seen in shur (intesity-free)
        total_time_loss += time_loss
        total_count += len(batch)

        total_pred_loss += pred_loss
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        optimizer.step()

    return total_time_loss/total_count, total_pred_loss/total_num_pred

def eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt, event_data):
    model.eval()
    total_time_loss = 0.0
    total_pred_loss = 0

    total_count = 0
    total_num_pred = 0
    with torch.no_grad():
        for  batch in tqdm(test_data, mininterval=2, desc='  - (Training)   ', leave=False):
            
            event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)

            enc_out, (prediction, mixture_enc) = model(event_type, event_time)

            pred_loss, pred_num_event = Utils.type_loss(prediction, event_type, pred_loss_func)

            gmm = Mix_Utils.get_inter_time_dist(model, opt, event_data, mixture_enc)
            time_loss = -Mix_Utils.log_probability(gmm, time_gap).mean()

            loss = time_loss + pred_loss
            #wandb.log({'batch_loss_test': loss})

            total_time_loss += time_loss
            total_count += len(batch)

            total_pred_loss += pred_loss
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        return total_time_loss/total_count, total_pred_loss/total_num_pred


def get_event_data(opt):
    #TODO this is on test data -> mean and variance is calculated on test data! Change it to train data
    pred_data, num_types = load_prediciton_data(opt)
    eos_test_ti = load_test_eos_data(opt)
    return EventData_Trim(pred_data, eos_test_ti, opt)
