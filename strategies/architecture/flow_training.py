from strategies.architecture.training_strategy import Training_Strategy

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils

### https://github.com/data-iitd/proactive/ ###
class FlowTraining(Training_Strategy):
    def train(self, model: nn.Module, training_data, test_data, optimizer: optim.Adam, scheduler, pred_loss_func, pred_loss_goal, opt):
        test_acc_list = []
        test_goal_list = []
        test_mae_list = []

        best_performance = 0
        for epoch_i in range(opt.epoch):
            epoch = epoch_i + 1
            print('[ Epoch', epoch, ']')

            train_event, train_type, train_goal, train_time = self.train_epoch(model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt)
            print('(Training) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, Itv. GPA: {goal: 8.5f}'.format(type=train_type, mae=train_time, goal=train_goal))
                
            test_event, test_type, test_goal, test_time = self.eval_epoch(model, test_data, pred_loss_func, pred_loss_goal, opt)
            print('(Testing) Acc: {type: 8.5f}, MAE: {mae: 8.5f}, GPA: {goal: 8.5f}\n'.format(type=test_type, mae=test_time, goal=test_goal))
            
            test_acc_list += [test_type]
            test_goal_list += [test_goal]
            test_mae_list += [test_time]
            print('Best ACC: {pred: 8.5f}, MAE: {mae: 8.5f}, GPA: {gpa: 8.5f}'.format(pred=max(test_acc_list), mae=min(test_mae_list), gpa=max(test_goal_list)))
            scheduler.step()

            current_performance = test_type + (1-test_time)
            if(current_performance > best_performance):
                print(f"improvement in epoch {epoch}")
                best_performance = current_performance
                model_state = model.state_dict()
                optim_state = optimizer.state_dict()
            print()
        return model_state, optim_state
        
    def train_epoch(self, model, training_data, optimizer, pred_loss_func, pred_loss_goal, opt):
        model.train()

        hawkes = False

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
            if hawkes:
                scale_time_loss = 100
                loss = event_loss + pred_loss + se / scale_time_loss
            else:
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

        return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_goal_rate / total_num_pred, mae
    
    def eval_epoch(self, model, test_data, pred_loss_func, pred_loss_goal, opt):
        model.eval()

        total_event_ll = 0
        total_time_se = 0
        total_event_rate = 0
        total_goal_rate = 0
        total_num_event = 0
        total_num_pred = 0
        total_seqs = 0

        with torch.no_grad():
            for batch in tqdm(test_data, mininterval=2, desc='  - (Validation) ', leave=False):
                event_time, time_gap, event_type, event_goal = map(lambda x: x.to(opt.device), batch)
                

                enc_out, prediction = model(event_type, event_time)

                event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
                ## What is event_logLik and non_event_loglik?
                
                event_loss = -torch.sum(event_ll - non_event_ll)
                _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)

                pred_goal, seq_num = Utils.pred_goal(prediction[2], event_goal)
                se = Utils.time_loss_flows(prediction[1], event_time)

                total_event_ll += -event_loss.item()
                total_time_se += se.item()
                total_event_rate += pred_num.item()
                total_goal_rate += pred_goal.item()
                total_seqs += seq_num.item()
                total_num_event += event_type.ne(Constants.PAD).sum().item()
                total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        mae = total_time_se / (total_num_pred)
        return total_event_ll / total_num_event, total_event_rate / total_num_pred, total_goal_rate / total_seqs, mae