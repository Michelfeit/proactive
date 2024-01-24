from myTransformer.Models import Transformer

import torch
import wandb
import pprint
from data_preparation import initial_dataloader_preparation
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils

### https://github.com/data-iitd/proactive/ ###

def train(config=None):
    with wandb.init(config=config):
        print()
        print("Training flows transformer...")
        print()
        config = wandb.config
        pprint.pprint(config)
        trainloader, testloader, num_types, num_goals = initial_dataloader_preparation(config)

        model = Transformer(
            num_types=num_types,
            num_goals=num_goals,
            d_model=config.d_model,
            d_rnn=config.d_rnn,
            d_inner=config.d_model * 4,
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
    
        test_acc_list = []
        test_goal_list = []
        test_mae_list = []

        best_loss = 10000
        for epoch_i in range(config.epochs):
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            train_event, train_type, train_goal, train_time = train_epoch(model, trainloader, optimizer, pred_loss_func, pred_loss_goal, config, epoch)
            print('(Training) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, Itv. GPA: {goal: 8.5f}'.format(type=train_type, mae=train_time, goal=train_goal))
            wandb.log({'accuracy_train': train_type, 'gpa_train': train_goal, 'mae_train': train_time})

            test_event, test_type, test_goal, test_time, test_loss = eval_epoch(model, testloader, pred_loss_func, pred_loss_goal, config)
            print('(Testing) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, GPA: {goal: 8.5f}\n'.format(type=test_type, mae=test_time, goal=test_goal))
            wandb.log({'accuracy_test': test_type, 'gpa_test': test_goal, 'mae_test': test_time})

            test_acc_list += [test_type]
            test_goal_list += [test_goal]
            test_mae_list += [test_time]
            print('Best ACC: {pred: 8.5f}, MAE: {mae: 8.5f}, GPA: {gpa: 8.5f}'.format(pred=max(test_acc_list), mae=min(test_mae_list), gpa=max(test_goal_list)))
            scheduler.step()

            wandb.log({'test_loss': test_loss})
            current_loss = test_loss
            if(current_loss > best_loss):
                print(f"improvement in epoch {epoch}")
                best_loss = current_loss
                model_state = model.state_dict()
                optim_state = optimizer.state_dict()
            print()
        return model_state, optim_state
    
def train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt, epoch):
    model.train()
    total_event_ll = 0
    total_time_se = 0
    total_event_rate = 0
    total_goal_rate = 0
    total_num_event = 0
    total_num_pred = 0

    ### tqdm is progress bar
    ### batch is a tensor
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):

        ### put batch element-wise on gpu
        event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time)

        # Likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # Type Prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)
        
        # Time Prediction
        se = Utils.time_loss_flows(prediction[1], event_time)

        # Goal Prediction
        goal_loss, pred_num_goal = Utils.goal_loss(prediction[2], event_goal, pred_loss_goal)

        # Scales to stabilize training
        scale_time_loss = 1
        scale_goal_loss = 10
        loss = event_loss + pred_loss + goal_loss/scale_goal_loss + se / scale_time_loss
        wandb.log({"sequence_loss": loss, "epoch": epoch})      
        loss.backward()
        optimizer.step()

        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_goal_rate += pred_num_goal.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    mae = total_time_se / total_num_pred

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_goal_rate / total_num_pred, mae

def eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt):
    model.eval()

    total_event_ll = 0
    total_time_se = 0
    total_event_rate = 0
    total_goal_rate = 0
    total_num_event = 0
    total_num_pred = 0
    total_seqs = 0

    total_loss = 0 # cumulative prediction loss

    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2, desc='  - (Validation) ', leave=False):
            event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)
            

            enc_out, prediction = model(event_type, event_time)

            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            
            event_loss = -torch.sum(event_ll - non_event_ll)
            pred_loss, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            goal_loss, pred_num_goal = Utils.goal_loss(prediction[2], event_goal, pred_loss_goal)

            pred_goal, seq_num = Utils.pred_goal(prediction[2], event_goal)
            se = Utils.time_loss_flows(prediction[1], event_time)

            scale_time_loss = 1
            scale_goal_loss = 10
            loss = event_loss + pred_loss + goal_loss/scale_goal_loss + se / scale_time_loss

            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_goal_rate += pred_goal.item()
            total_seqs += seq_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

            total_loss += loss.item()

    mae = total_time_se / (total_num_pred)
    test_loss = total_loss/total_num_event
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_goal_rate / total_seqs, mae, test_loss