import pickle
from process import get_dataloader
from trim_process import get_trim_dataloader

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

# def load_all_data(opt):
#     train_data, num_types, num_goals = _load_data(opt.data + 'train.pkl', 'train')
#     test_data, _, _ = _load_data(opt.data + 'test.pkl', 'test')
#     pred_data = test_data

#     return train_data, test_data, pred_data, num_types, num_goals