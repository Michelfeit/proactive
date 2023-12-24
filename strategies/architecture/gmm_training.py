from myTransformer.Models import Transformer

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils
import myTransformer.Mixture_Utils as Mix_Utils

from strategies.architecture.training_strategy import Training_Strategy

class GMM_Training(Training_Strategy):
    def train(self, model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt):
        return super().train(model, training_data, test_data, optimizer, scheduler, pred_loss_func, pred_loss_goal, opt)
    
    def train_epoch(self, model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt):
        """ Epoch operation in training phase. """
        model.train()
        for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
            opt.zero_grad()
            event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)
            enc_out, (prediction, mixture_enc) = model(event_type, event_time)
            gmm = Mix_Utils.get_inter_time_dist(mixture_enc)
            loss = -Mix_Utils.log_probability(gmm, time_gap).mean()
            loss = -model.log_prob(batch).mean()
            loss.backward()
            opt.step()




        return super().train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt)
    
    def eval_epoch(self, model, test_data, pred_loss_func, pred_loss_goal, opt):
        return super().eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt)