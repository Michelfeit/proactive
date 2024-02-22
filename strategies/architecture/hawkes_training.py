from myTransformer.Models import Transformer

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils

from strategies.architecture.training_strategy import Training_Strategy

### https://github.com/SimiaoZuo/Transformer-Hawkes-Process/tree/master ###
class HawkesTraining(Training_Strategy):
    def train(self, model: nn.Module, training_data, test_data, optimizer: optim.Adam, scheduler, pred_loss_func, pred_loss_goal, opt):
        """ Start training. """
        valid_event_losses = []  # validation log-likelihood
        valid_pred_losses = []  # validation event type prediction accuracy
        valid_rmse = []  # validation event time prediction RMSE

        best_performance = 0
        for epoch_i in range(opt.epoch):
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            start = time.time()
            train_event, train_type, train_time = self.train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt)
            print('  - (Training)    loglikelihood: {ll: 8.5f}, '
                'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                'elapse: {elapse:3.3f} min'
                .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

            start = time.time()
            valid_event, valid_type, valid_time, mae = self.eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt)
            print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
                'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
                'elapse: {elapse:3.3f} min'
                .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))
            print("MAE:", mae)
            valid_event_losses += [valid_event]
            valid_pred_losses += [valid_type]
            valid_rmse += [valid_time]
            print('  - [Info] Maximum ll: {event: 8.5f}, '
                'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}\n'
                .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))
            
            current_performance = valid_type + (1-valid_time)
            if(current_performance > best_performance):
                print(f"improvement in epoch {epoch}")
                best_performance = current_performance
                model_state = model.state_dict()
                optim_state = optimizer.state_dict()
            print()
            # # logging
            # with open(opt.log, 'a') as f:
            #     f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
            #             .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))
            scheduler.step()  
        return model_state, optim_state
             
    
    def train_epoch(self, model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt):
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
    
    def eval_epoch(self, model, test_data, pred_loss_func, pred_loss_goal, opt):
        """ Epoch operation in evaluation phase. """
        model.eval()

        total_event_ll = 0  # cumulative event log-likelihood
        total_time_se = 0  # cumulative time prediction squared-error
        total_event_rate = 0  # cumulative number of correct prediction
        total_num_event = 0  # number of total events
        total_num_pred = 0  # number of predictions

        total_time_ae = 0
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
                _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
                se = Utils.time_loss_hawkes(prediction[1], event_time)

                ae = Utils.time_loss_flows(prediction[1], event_time)

                """ note keeping """
                total_event_ll += -event_loss.item()
                total_time_se += se.item()
                total_time_ae += ae.item()
                total_event_rate += pred_num.item()
                total_num_event += event_type.ne(Constants.PAD).sum().item()
                total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        rmse = np.sqrt(total_time_se / total_num_pred)
        mae = total_time_ae / (total_num_pred)
        return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse, mae