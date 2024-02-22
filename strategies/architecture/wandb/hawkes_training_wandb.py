from myTransformer.Models import Transformer

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pprint
import wandb
from tqdm import tqdm
from data_preparation import initial_dataloader_preparation
from data_preparation import get_prediction_loader, load_prediciton_data, load_test_eos_data

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils

def train(config=None):
    with wandb.init(config=config):
        print("Training hawkes transformer...")
        config = wandb.config
        pprint.pprint(config)
        trainloader, testloader, num_types, num_goals = initial_dataloader_preparation(config)

        model = Transformer(
            num_types=num_types,
            num_goals=num_goals,
            d_model=config.n_head * config.d_k,
            d_rnn=config.d_rnn,
            d_inner=config.n_head * config.d_k * 4,
            n_layers=config.n_layers,
            n_head=config.n_head,
            d_k=config.d_k,
            d_v=config.d_k,
            dropout=config.dropout,
            activation=config.activation,
            hawkes=True
        )
        model.to(torch.device(config.device))

        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), config.learning_rate, betas=(0.9, 0.999), eps=1e-05)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

        if config.smooth > 0:
            pred_loss_func = Utils.LabelSmoothingLoss(config.smooth, num_types, ignore_index=-1)
            pred_loss_goal = Utils.LabelSmoothingLoss(config.smooth, num_goals, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            pred_loss_goal = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            
        """ Start training. """
        valid_event_losses = []  # validation log-likelihood
        valid_pred_losses = []  # validation event type prediction accuracy
        valid_rmse = []  # validation event time prediction RMSE

        best_loss = 10000
        for epoch_i in range(config.epochs):
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            start = time.time()
            train_event, train_type, train_time = train_epoch(model, trainloader, optimizer, pred_loss_func, pred_loss_goal, config, epoch)
            print('  - (Training)    loglikelihood: {ll: 8.5f}, '
                'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                'elapse: {elapse:3.3f} min'
                .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))
            
            wandb.log({'accuracy_train': train_type, 'loglikelihood_train': train_event, 'rmse_train': train_time})

            start = time.time()
            valid_event, valid_type, valid_time, test_loss = eval_epoch(model, testloader, pred_loss_func, pred_loss_goal, config)
            print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
                'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                'elapse: {elapse:3.3f} min'
                .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))
            
            wandb.log({'accuracy_test': valid_type, 'loglikelihood_test': valid_event, 'rmse_test': valid_time})

            valid_event_losses += [valid_event]
            valid_pred_losses += [valid_type]
            valid_rmse += [valid_time]
            print('  - [Info] Maximum ll: {event: 8.5f}, '
                'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}\n'
                .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))
            
            wandb.log({'test_loss': test_loss})
            
            current_loss = test_loss
            if(current_loss < best_loss):
                print(f"improvement in epoch {epoch}")
                best_loss = current_loss
                model_state = model.state_dict()
                optim_state = optimizer.state_dict()
            print()
            scheduler.step()  
        return model_state, optim_state
             
    
def train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt, epoch):
    """ Epoch operation in training phase. """
    model.train()
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                    desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss_hawkes(prediction[1], event_time)

        # Goal Prediction
        goal_loss, pred_num_goal = Utils.goal_loss(prediction[2], event_goal, pred_loss_goal)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 1
        loss = event_loss + pred_loss + se / scale_time_loss
        wandb.log({"sequence_loss": loss, "epoch": epoch})
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse
    
def eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt):
    """ Epoch operation in evaluation phase. """
    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_time_se_for_loss = 0 # cumulative time prediction squared-error with scaling
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    total_loss = 0 # cumulative prediction loss
    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,
                        desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            event_loss = -torch.sum(event_ll - non_event_ll)
            pred_loss, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss_hawkes(prediction[1], event_time)

            scale_time_loss = 100
            loss = event_loss + pred_loss + se / scale_time_loss

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_time_se_for_loss += se.item() / scale_time_loss
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]
            total_loss += loss.item()

            rmse = np.sqrt(total_time_se / total_num_pred)

        test_loss = total_loss/total_num_event

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, test_loss