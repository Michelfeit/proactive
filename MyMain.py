import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils
from process import get_dataloader
from trim_process import get_trim_dataloader
from myTransformer.Models import Transformer
import pdb

MODEL_PATH = "trainedModels\\transformer01.pth.tar"

def prepare_dataloader(opt):
    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            
            num_goals = data['dim_goals']

            data = data[dict_name]
            return data, int(num_types), int(num_goals)

    print('Loading All Datasets...')
    train_data, num_types, num_goals = load_data(opt.data + 'train.pkl', 'train')
    test_data, _, _ = load_data(opt.data + 'test.pkl', 'test')

   
    

    print(f"Number of types: {num_types}")
    print(f"Number of goals: {num_goals}")

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
     # add third type of data that trims the test data down to a given percentile
    pred_data = get_trim_dataloader(test_data, opt.batch_size, shuffle=False, alpha=0.3)
    return trainloader, testloader, num_types, num_goals

def train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt):
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
        se = Utils.time_loss(prediction[1], event_time)

        # Goal Prediction
        goal_loss, pred_num_goal = Utils.goal_loss(prediction[2], event_goal, pred_loss_goal)

        # Scales to stabilize training
        scale_time_loss = 1
        scale_goal_loss = 10
        loss = event_loss + pred_loss + goal_loss/scale_goal_loss + se / scale_time_loss
        loss.backward()

        optimizer.step()

        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_goal_rate += pred_num_goal.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    mae = total_time_se / total_num_pred
    #
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

    with torch.no_grad():
        i = 0
        for batch in tqdm(test_data, mininterval=2, desc='  - (Validation) ', leave=False):
            event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)
            ## a batch is a subset of the testdata. our batchsize is 4. This way, the batch consists of 4 event_types and event_times as specified in
            ## the datasets. Also so list of types and times is filled at the end with zeros to fit action with the most amount of events per batch.

            ## For the first batch, 
            ## - the event_type tensor looks like this:
            ##      [[ 1, 19, 20, 22,  1,  0,  0,  0],
            ##       [ 1, 19, 20, 22,  1,  0,  0,  0],
            ##       [ 1, 20, 22,  1,  0,  0,  0,  0],
            ##       [ 1, 16, 21, 19, 23, 20, 22,  1]]
            ## - the event_times look like this:
            ##      [[0.0000, 0.0217, 0.2955, 1.2562, 1.7056, 0.0000, 0.0000, 0.0000],
            ##       [0.0000, 0.0640, 0.1787, 0.7769, 0.9215, 0.0000, 0.0000, 0.0000],
            ##       [0.0000, 0.1374, 0.4742, 0.5837, 0.0000, 0.0000, 0.0000, 0.0000],
            ##       [0.0000, 0.0475, 0.1694, 0.2562, 0.4287, 0.4587, 1.0434, 1.2531]]

            enc_out, prediction = model(event_type, event_time)

            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
            ## What is event_logLik and non_event_loglik?
            
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            pred_goal, seq_num = Utils.pred_goal(prediction[2], event_goal)
            se = Utils.time_loss(prediction[1], event_time)

            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_goal_rate += pred_goal.item()
            total_seqs += seq_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    mae = total_time_se / (total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_goal_rate / total_seqs, mae

# the original train method used in the proactive paper
def train(model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
    test_acc_list = []
    test_goal_list = []
    test_mae_list = []
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_goal, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt)
        print('(Training) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, Itv. GPA: {goal: 8.5f}'.format(type=train_type, mae=train_time, goal=train_goal))

        print(f"(Training) Acc: {train_type:8.5f}")
              
        start = time.time()
        test_event, test_type, test_goal, test_time = eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt)
        print('(Testing) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, GPA: {goal: 8.5f}'.format(type=test_type, mae=test_time, goal=test_goal))

        test_acc_list += [test_type]
        test_goal_list += [test_goal]
        test_mae_list += [test_time]
        print('Best ACC: {pred: 8.5f}, MAE: {mae: 8.5f}, GPA: {gpa: 8.5f}'.format(pred=max(test_acc_list), mae=min(test_mae_list), gpa=max(test_goal_list)))

        scheduler.step()

def eval_by_gupta(model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
    test_acc_list = []
    test_goal_list = []
    test_mae_list = []

    # for epoch_i in range(opt.epoch):
    #     epoch = epoch_i + 1
    # print('[ Epoch', epoch, ']')

    # is this necessary?
    optimizer.step()

    start = time.time()
    test_event, test_type, test_goal, test_time = eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt)
    print('(Testing) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, GPA: {goal: 8.5f}'.format(type=test_type, mae=test_time, goal=test_goal))

    test_acc_list += [test_type]
    test_goal_list += [test_goal]
    test_mae_list += [test_time]
    print('Best ACC: {pred: 8.5f}, MAE: {mae: 8.5f}, GPA: {gpa: 8.5f}'.format(pred=max(test_acc_list), mae=min(test_mae_list), gpa=max(test_goal_list)))

    scheduler.step()

# training the model without starting the evaluation on test set afterwards
def train_without_eval(model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_goal, train_time = train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt)
        print('(Training) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, Itv. GPA: {goal: 8.5f}'.format(type=train_type, mae=train_time, goal=train_goal))

#python MyMain.py -data data/Breakfast/ -batch 4 -n_head 4 -n_layers 4 -d_model 64 -d_inner 256 -epoch 1 -proactive=False
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', required=True)
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
    parser.add_argument('-proactive', type=boolean_string, default= 'True')
    opt = parser.parse_args()

    opt.device = torch.device('cuda')
    trainloader, testloader, num_types, num_goals = prepare_dataloader(opt)

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

    # when proactive flag is set, training and evaluation occurs as provided by the paper 'proactive'
    if(opt.proactive):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)
        return
    
    # when proactive flag is set to 'False', the alternative evaluation is executed, 
    # check for a trained modell at MODEL_PATH.
    if Path(MODEL_PATH).is_file():
        print("Trained transformer found. Initializing model parameters...")

    else:
        print("No model found. Initiating training...")

        # a model is trained and saved to evaluate at a later time
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        train_without_eval(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH)

    save_state = torch.load(MODEL_PATH)
    model.load_state_dict(save_state['model_state_dict'])
    optimizer.load_state_dict(save_state['optimizer_state_dict'])
    print("start evaluation:")
    eval_by_gupta(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    main()