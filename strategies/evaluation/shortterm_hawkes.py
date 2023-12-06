from strategies.evaluation.evaluation_strategy import Evaluation_Strategy

import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils
from data_preparation import initial_dataloader_preparation

class Shortterm_Hawkes_Strategy(Evaluation_Strategy):
    def evaluate(self, model, optimizer, scheduler, opt):
        _, testloader , num_types, num_goals = initial_dataloader_preparation(opt)

        if opt.smooth > 0:
            pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        start = time.time()

        valid_event, valid_type, valid_time = self.eval_epoch(model, testloader, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
            'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
            'elapse: {elapse:3.3f} min'
            .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))
    
    def eval_epoch(self, model, test_data, pred_loss_func, opt):
        """ Epoch operation in evaluation phase. """
        model.eval()

        total_event_ll = 0  # cumulative event log-likelihood
        total_time_se = 0  # cumulative time prediction squared-error
        total_event_rate = 0  # cumulative number of correct prediction
        total_num_event = 0  # number of total events
        total_num_pred = 0  # number of predictions
        with torch.no_grad():
            for batch in tqdm(test_data, mininterval=2,
                            desc='  - (Validation) ', leave=False):
                """ prepare data """
                event_time, time_gap, event_type, _ = map(lambda x: x.to(opt.device), batch)

                """ forward """
                enc_out, prediction = model(event_type, event_time)

                """ compute loss """
                event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type)
                event_loss = -torch.sum(event_ll - non_event_ll)
                _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
                se = Utils.time_loss_hawkes(prediction[1], event_time)

                """ note keeping """
                total_event_ll += -event_loss.item()
                total_time_se += se.item()
                total_event_rate += pred_num.item()
                total_num_event += event_type.ne(Constants.PAD).sum().item()
                total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        rmse = np.sqrt(total_time_se / total_num_pred)
        return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse