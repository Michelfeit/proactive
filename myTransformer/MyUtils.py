import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_next_type_prediction(prediction, data):
    pred_type = torch.max(prediction, dim=-1)[1]
    preds = []
    j = 0
    for sequence in data:
        i = 0
        for action in sequence:
            if(action == 0):
                break
            i += 1
        preds.append(pred_type[j][i-1])
        j+=1
    next_action = torch.tensor(preds) + 1
    return next_action, pred_type

def get_next_time_prediction(prediction, data):
    prediction.squeeze_(-1)
    preds = []
    j = 0
    for sequence in data:
        i = 0
        for action in sequence[1:]:
            if(action == 0):
                break
            i += 1
        preds.append(prediction[j][i])
        
        j+=1
    next_time = torch.tensor(preds)
    return next_time, prediction

def get_next_goal_prediciton(prediction, types):
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    pred_type = pred_type.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()

    preds = []
    trs = []
    for i in range(len(truth)):
        id_ = np.argwhere(truth[i] == -1)
        if len(id_) == 0:
            preds.append(pred_type[i][-1])
        else:
            preds.append(pred_type[i][id_[0][0]-1])
        trs.append(truth[i][0])
    return preds

