import numpy as np
import torch
import torch.utils.data
from transformer import Constants
from process import EventData

class EventData_Trim(EventData):
    def __init__(self, data, alpha):
        super().__init__(data)
        self.alpha = alpha
        # get a list if indeces that stand for the index to be trimmed for each sequence
        self.alpha_indeces = self.get_alpha_trimmed_indeces()
        time = []
        time_gap = []
        event_type = []
        event_goal = []
        i = 0
        
        # generate lists of action sequences trimmed to the desired length given by indeces
        for index in self.alpha_indeces:
            time.append(self.time[i][:index])
            time_gap.append(self.time_gap[i][:index])
            event_type.append(self.event_type[i][:index])
            event_goal.append(self.event_goal[i][:index])
            i += 1
        
        self.trim_time = time
        self.trim_time_gap = time_gap
        self.trim_event_type = event_type
        self.trim_event_goal = event_goal

        print("EVENTDATA")
        print(self.time[48])
        print(self.trim_time[48])

    def concat_predictions(self, pred_time_gap, pred_event_type, pred_event_goal):
        if not pred_time_gap:
           return
        assert(len(pred_time_gap) == len(pred_event_type) == len(pred_event_goal) == self.length)
        for i in range(self.length):
            self.trim_time[i].append(self.trim_time[i][-1] + pred_time_gap[i])
            self.trim_time_gap[i].append(pred_time_gap[i])
            self.trim_event_type[i].append(pred_event_type[i])
            self.trim_event_goal[i].append(pred_event_goal[i])

    # provides the index (excluded) of the last action within the alpha-percentile window of each sequence
    def get_alpha_trimmed_indeces(self):
        indeces = []
        for seq in self.time:
            limit = seq[-1] * self.alpha
            index = 0
            for i in range(seq.__len__()):
                if(seq[i] > limit):
                    index = i
                    break
            indeces.append(index)
        return indeces
    
    def get_trim_data(self):
        return self.trim_time, self.trim_time_gap, self.trim_event_type, self.trim_event_goal
    
    #for a given beta, get truths trimmed down to alpha + alpha
    def get_beta_trimmed_truths(self, beta):
        
        indeces = []
        for seq in self.time:
            limit = seq[-1] * (self.alpha + beta)
            index = 0
            for i in range(seq.__len__()):
                if(seq[i] > limit):
                    index = i
                    break
            indeces.append(index)
        time = []
        time_gap = []
        event_type = []
        event_goal = []
        i = 0
        # generate lists of action sequences trimmed to the desired length given by indeces
        for index in indeces:
            time.append(self.time[i][:index])
            time_gap.append(self.time_gap[i][:index])
            event_type.append(self.event_type[i][:index])
            event_goal.append(self.event_goal[i][:index])
            i += 1
        return time, time_gap, event_type, event_goal
        
    # trims lists of data down, given a list of indeces(excluded)
    #def trim_data(indeces):


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #trims = (self.trim_time[idx], self.trim_time_gap[idx], self.trim_event_type[idx], self.trim_event_goal[idx])
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.event_goal[idx], self.trim_time[idx], self.trim_time_gap[idx], self.trim_event_type[idx], self.trim_event_goal[idx]

def pad_time(insts):
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.float32)

def pad_type(insts):
    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    return torch.tensor(batch_seq, dtype=torch.long)

def collate_trim_fn(insts):
    time, time_gap, event_type, event_goal, trim_time, trim_gap, trim_event_type, trim_event_goal = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    event_goal = pad_type(event_goal)

    trim_time = pad_time(trim_time)
    trim_gap = pad_time(trim_gap)
    trim_event_type = pad_type(trim_event_type)
    trim_event_goal = pad_type(trim_event_goal)
    i = 1
    return time, time_gap, event_type, event_goal, trim_time, trim_gap, trim_event_type, trim_event_goal

def get_trim_dataloader(ds, batch_size, shuffle=True, alpha= .3):
    #pass ds as argument in order to be able to concat data
    # ds = EventData_Trim(data, alpha)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_trim_fn,
        shuffle=shuffle
    )
    return dl