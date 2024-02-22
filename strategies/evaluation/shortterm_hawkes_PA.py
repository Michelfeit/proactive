from strategies.evaluation.evaluation_strategy import Evaluation_Strategy

import torch
import torch.nn as nn
from tqdm import tqdm

import myTransformer.Constants as Constants
import myTransformer.Utils as Utils
from data_preparation import initial_dataloader_preparation

MODEL_PATH = "trainedModels\\transformer50softplus.pth.tar"
LONGEST_TEST_ACTION_SEQUENCE = 20
ALPHA = .3
LIST_OF_BETA_VALUES = [0.1, 0.2, 0.3, 0.5]
BREAKFAST_FRAME_RATE = 15

class Shortterm_Hawkes_Strategy(Evaluation_Strategy):
    def evaluate(self, model, optimizer, scheduler, opt):
        _, testloader , num_types, num_goals = initial_dataloader_preparation(opt)

        if opt.smooth > 0:
            pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
        else:
            pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

        test_event, test_type, test_time = self.eval_epoch(model, testloader, pred_loss_func, opt)
        print('(Testing) Acc: {type: 8.5f}, MAE: {mae: 8.5f}'.format(type=test_type, mae=test_time))

    def eval_epoch(self, model, test_data, pred_loss_func, opt):
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

                se = Utils.time_loss_flows(prediction[1], event_time)

                total_event_ll += -event_loss.item()
                total_time_se += se.item()
                total_event_rate += pred_num.item()
                # total_goal_rate += pred_goal.item()
                # total_seqs += seq_num.item()
                total_num_event += event_type.ne(Constants.PAD).sum().item()
                total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

        mae = total_time_se / (total_num_pred)
        return total_event_ll / total_num_event, total_event_rate / total_num_pred, mae
