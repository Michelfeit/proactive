import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_next_type_prediction(prediction, data):
    pred_type = torch.max(prediction, dim=-1)[1]
    sampling = False
    # softmax on ptredicitons
    # torch.sample
    if(sampling):
        type_softmax  = F.softmax(pred_type, dim=0)
        sampled_index = torch.multinomial(type_softmax, 1).item()
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

def reverse_time_normalization():
    in_file = open('data/Breakfast' + '/train_ti.txt', 'r')
    times = [[float(y) for y in x.strip().split()] for x in in_file]

    min = 10000
    max = 0
    scale = 10
    # taken form develop_dumbs
    for i in range(len(times)):
        for j in range(len(times[i])):
            val = float(times[i][j])
            if(val > max):
                max = val
            if(val < min):
                min = val
    # function that reverses the normalization that is applied on the event_times in develop_dumps.py
    return lambda x: (x/scale * (max - min) ) - min

def time_normalization():
    in_file = open('data/Breakfast' + '/train_ti.txt', 'r')
    times = [[float(y) for y in x.strip().split()] for x in in_file]

    min = 10000
    max = 0
    scale = 10
    # taken form develop_dumbs
    for i in range(len(times)):
        for j in range(len(times[i])):
            val = float(times[i][j])
            if(val > max):
                max = val
            if(val < min):
                min = val
    # function that produces the normalization that is applied on the event_times in develop_dumps.py
    return lambda x: scale * (x - min)/(max - min)

# right function
def truth_to_frame_data(times, types):
    if 0 in types:
        index, types = _adjust_types(types)
        times = times[:index + 1]
    else:
        times.append(times[-1])

    reverse_norm = reverse_time_normalization()
    orig = list(map(reverse_norm, times))
    orig = list(map(lambda x: int(round(x)), orig))
    orig = orig[1:]
    start = 0
    in_frames = [1]

    for index, action in enumerate(types):
        ran = range(start, orig[index])
        if len(ran) == 0:
            ran = range(0,10)
        for i in ran:
            in_frames.append(action)
        start = orig[index]
    return in_frames

def _adjust_types(types):
    index = types.index(0)
    types = types[:index]
    return index, types

