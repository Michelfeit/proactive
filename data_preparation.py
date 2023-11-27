import pickle
from process import get_dataloader
from trim_process import get_trim_dataloader
import sys

def initial_dataloader_preparation(opt):
    print('Loading All Datasets...')
    train_data, num_types, num_goals = _load_data(opt.data + 'train.pkl', 'train')
    test_data, _, _ = _load_data(opt.data + 'test.pkl', 'test')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=False)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
     # add third type of data that trims the test data down to a given percentile
    return trainloader, testloader, num_types, num_goals

def _load_data(name, dict_name):
    with open(name, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
        num_types = data['dim_process']
        
        num_goals = data['dim_goals']

        data = data[dict_name]
        return data, int(num_types), int(num_goals)
    
# def get_dataloaders(opt, train_data, test_data, pred_data, num_types, num_goals):
#     trainloader = get_dataloader(train_data, opt.batch_size, shuffle=False)
#     testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
#     pred_data = get_trim_dataloader(test_data, opt.batch_size, shuffle=False, alpha=0.3)

#     return trainloader, testloader, pred_data, num_types, num_goals

def get_prediction_loader(opt, data):
    pred_data = get_trim_dataloader(data, opt.batch_size, shuffle=False, alpha=0.3)
    return pred_data
    
def load_prediciton_data(opt):
    pred_data, _, _ = _load_data(opt.data + 'test.pkl', 'test')
    return pred_data

def reverse_scale(opt):
    in_file = open(opt.data+'/train_ti.txt', 'r')
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
    print(min, max)
    # function that reverses the normalization that is applied on the event_times in develop_dumps.py
    return lambda x: (x/scale * (max - min) ) - min