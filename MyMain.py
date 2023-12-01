import argparse
import math
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

import myTransformer.Constants as Constants
import myTransformer.MyUtils as MyUtils
import myTransformer.Utils as Utils
from data_preparation import initial_dataloader_preparation
from data_preparation import get_prediction_loader, load_prediciton_data, load_test_eos_data
from trim_process import EventData_Trim, get_trim_dataloader
# from process import get_dataloader
# from trim_process import get_trim_dataloader, EventData_Trim
from myTransformer.Models import Transformer
import pdb

MODEL_PATH = "trainedModels\\transformer50relu.pth.tar"
LONGEST_TEST_ACTION_SEQUENCE = 20
ALPHA = .3
LIST_OF_BETA_VALUES = [0.1, 0.2, 0.3, 0.5]
BREAKFAST_FRAME_RATE = 15

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

def predict(model, optimizer, scheduler, opt):
    pred_data, num_types = load_prediciton_data(opt)
    eos_test_ti = load_test_eos_data(opt)

    # trim_event_data encapsules trimmed down versions of action sequences contained in the test.pkl
    # during this prediction, the eventdata ought to be concatenated with predictions by the model
    trim_event_data = EventData_Trim(pred_data, eos_test_ti, ALPHA)
    
    pred_time = []
    pred_time_gap = []
    pred_event_type = []
    pred_event_goal = []

    # function that turns times back into seconds
    times_in_seconds = MyUtils.reverse_time_normalization()

    predicting = True
    model.eval()
    with torch.no_grad():
        i = 0
        while(predicting):
            print("step:", i)
            i += 1
            # add the gap times to the last time when concatting
            trim_event_data.concat_predictions(pred_time_gap, pred_event_type, pred_event_goal)
            predictionloader = get_prediction_loader(opt, trim_event_data)
            pred_time = []
            pred_time_gap = []
            pred_event_type = []
            pred_event_goal = []
            ## for debugging purposes
            predicting = False
            j = 0
            
            for batch in predictionloader:
                event_time, time_gap, event_type, event_goal, trim_time, _ , trim_type, trim_goal = map(lambda x: x.to(opt.device), batch)
                # find out if a sequence ended
                sequence_ended = [False] * len(trim_type)
                for seq in range(len(trim_type)):
                    if 1 in trim_type[seq][1:]:
                        sequence_ended[seq] = True
                    else:
                        # once a sequence is found that has not yet ended, flag is set and predicition keeps on looping
                        predicting = True

                enc_out, prediction = model(trim_type, trim_time)
                
                # get next event predicitons  
                pred_types, all_types  = MyUtils.get_next_type_prediction(prediction[0], trim_type)
                # event times
                pred_times, all_times = MyUtils.get_next_time_prediction(prediction[1], trim_time)
                #print(all_times)
                # goals
                pred_goals = MyUtils.get_next_goal_prediciton(prediction[2],trim_goal)
                
                # whenever a sequence in that batch ended, swap prediction with a zero
                for seq in range(len(trim_type)):
                    if(sequence_ended[seq]):
                        pred_types[seq] = torch.tensor(0)
                        pred_times[seq] = torch.tensor(0)
                        pred_goals[seq] = torch.tensor(0)

                pred_event_type += [element.item() for element in pred_types]
                pred_time_gap += [element.item() for element in pred_times]
                pred_event_goal += [element.item() for element in pred_goals]
                j +=1
            # LONGEST_TEST_ACTION_SEQUENCE
            if(i > LONGEST_TEST_ACTION_SEQUENCE):
                predicting = False
        print("DONE")
        eval_prediction(trim_event_data, num_types)

def eval_prediction(trim_event_data, num_types):
    print("Start evaluation:")
    for beta in range(len(LIST_OF_BETA_VALUES)):
        # the predicitons
        pred_time, pred_time_gap, pred_type, pred_goal = trim_event_data.get_trim_data()
        
        # the ground truth
        truths_time, truths_time_gap, truths_type, truths_goal = trim_event_data.get_truths()

        num_correct = [0] * num_types
        num_total = [0] * num_types

        num_skips = 0
        for i in range(len(pred_type)):
            if any(x < 0 for x in pred_time_gap[i]):
                num_skips += 1
                continue
            #convert to frames
            truths_framedata = MyUtils.truth_to_frame_data(truths_time[i],truths_type[i])
            preds_framedata = MyUtils.truth_to_frame_data(pred_time[i],pred_type[i])
            #print()
            # first frame of the prediciton
            alpha_index = math.ceil(len(truths_framedata) * ALPHA) + 1
            # first frame outside of prediciton window
            beta_index = math.ceil(len(truths_framedata) * (ALPHA + LIST_OF_BETA_VALUES[beta])) + 1
            len_dif = len(truths_framedata) - len( preds_framedata)
            if(len_dif > 0):
                preds_framedata.extend([preds_framedata[-1]] * len_dif)

            tru =  truths_framedata[alpha_index:beta_index]
            pred = preds_framedata[alpha_index:beta_index]

            zips = zip(tru, pred)
            for pair in zips:
                if(pair[0] == pair[1]):
                    num_correct[pair[0] - 1] += 1
                num_total[pair[0] - 1] += 1

        MOC_list = [-1] * num_types 
        for index in range(num_types):
            if(num_total[index] != 0):
                MOC_list[index] = (num_correct[index]/num_total[index]) * 100

        print(LIST_OF_BETA_VALUES[beta], MOC_list)

        count = 0
        percents = 0
        for entry in MOC_list:
            if(entry != -1):
                count += 1
                percents += entry
        print(f"Mean over all classes{LIST_OF_BETA_VALUES[beta]}:", percents/count)

        print("Skipped:" , num_skips)
        
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
            i += 1
            # print(i)
            event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)
            

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

def eval_by_gupta(model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
    test_acc_list = []
    test_goal_list = []
    test_mae_list = []

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

# python MyMain.py -data data/Breakfast/ -batch 4 -n_head 4 -n_layers 4 -d_model 64 -d_inner 256 -epoch 1 -proactive=False
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

    model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal = prepare_proactive(opt)
    # when proactive flag is set, training and evaluation occurs as provided by the paper 'proactive'
    if(opt.proactive):
        
        train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)
        return
    # in here the logic of prediction future events and evaluating their accuracy in regards to metrics given by
    # "A Survey on Deep Learning Techniques for Action Anticipation" is contained.
    run_action_prediciton(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)

def prepare_proactive(opt):
    trainloader, testloader, num_types, num_goals = initial_dataloader_preparation(opt)
    # train_data, test_data, pred_data, num_types, num_goals = load_all_data(opt)
    # get_dataloaders(opt, train_data, test_data, pred_data, num_types, num_goals)
 
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

    return model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal

def run_action_prediciton(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
    # check for a trained modell at MODEL_PATH.
    if Path(MODEL_PATH).is_file():
        print("Trained transformer found. Initializing model parameters...")
        save_state = torch.load(MODEL_PATH)
        model.load_state_dict(save_state['model_state_dict'])
        optimizer.load_state_dict(save_state['optimizer_state_dict'])
    else:
        print("No model found. Initiating training...")
        # a model is trained and saved to evaluate at a later time
        train_without_eval(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH)
        
    print("Predicting...")
    predict(model, optimizer, scheduler, opt)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    main()