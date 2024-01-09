from myTransformer.Models import Transformer

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_preparation import get_prediction_loader, load_prediciton_data, load_test_eos_data

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils
import myTransformer.Mixture_Utils as Mix_Utils

from strategies.architecture.training_strategy import Training_Strategy
from trim_process import EventData_Trim

class GMM_Training(Training_Strategy):
    def train(self, model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
        test_acc_list = []
        test_goal_list = []
        test_mae_list = []

        best_loss = 0

        event_data = self.get_event_data(opt)

        for epoch_i in range(opt.epoch):
            print('[ Epoch', epoch_i, ']')
            self.train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt, event_data)
            current_loss = self.eval_epoch(model, training_data, pred_loss_func, pred_loss_goal, opt, event_data)
            print("Test loss:", current_loss)
            if current_loss < best_loss:
                print("New best found!")
                best_loss = current_loss
                model_state = model.state_dict()
                optim_state = optimizer.state_dict()
                
        print("Best loss is:", best_loss)
        return model_state, optim_state
    
    def train_epoch(self, model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt, event_data):
        """ Epoch operation in training phase. """
        model.train()
        
        for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
            optimizer.zero_grad()

            event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)

            enc_out, (prediction, mixture_enc) = model(event_type, event_time)

            gmm = Mix_Utils.get_inter_time_dist(model, event_data, mixture_enc)
            temp_loss = -Mix_Utils.log_probability(gmm, time_gap)
            loss = temp_loss.mean()

            loss.backward()
            optimizer.step()
           # print("Training loss:", loss.item())
    
    def eval_epoch(self, model, test_data, pred_loss_func, pred_loss_goal, opt, event_data):
        model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for  batch in tqdm(test_data, mininterval=2, desc='  - (Training)   ', leave=False):
                
                event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)

                enc_out, (prediction, mixture_enc) = model(event_type, event_time)
                gmm = Mix_Utils.get_inter_time_dist(model, event_data, mixture_enc)
                total_loss += -Mix_Utils.log_probability(gmm, time_gap).sum()

                total_count += len(batch)
            return total_loss/total_count

    
    def get_event_data(self, opt):
        #TODO this is on test data -> mean and varianz is calculated on test data! Change it to train data
        pred_data, num_types = load_prediciton_data(opt)
        eos_test_ti = load_test_eos_data(opt)
        return EventData_Trim(pred_data, eos_test_ti, opt)
