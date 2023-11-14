import numpy as np
import pdb
import torch
import torch.utils.data
from transformer import Constants

class EventData_Trim(torch.utils.data.Dataset):
    def __init__(self, data, alpha):
        self.alpha = alpha
        self.time = [[elem['time_since_start'] for elem in inst] for inst in data]
        self.time_gap = [[elem['time_since_last_event'] for elem in inst] for inst in data]
        self.event_type = [[elem['type_event'] + 1 for elem in inst] for inst in data]
        self.event_goal = [[elem['type_goal'] for elem in inst] for inst in data]
        self.length = len(data)

        # get a list if indeces that stand for the index to be trimmed for ewach sequence
        indeces = []
        for seq in self.time:
            limit = seq[-1] * self.alpha
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
        
        self.time = time
        self.time_gap = time_gap
        self.event_type = event_type
        self.event_goal = event_goal

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.time[idx], self.time_gap[idx], self.event_type[idx], self.event_goal[idx]

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
    time, time_gap, event_type, event_goal = list(zip(*insts))
    time = pad_time(time)
    time_gap = pad_time(time_gap)
    event_type = pad_type(event_type)
    event_goal = pad_type(event_goal)
    return time, time_gap, event_type, event_goal

def get_trim_dataloader(data, batch_size, shuffle=True, alpha= .3):
    ds = EventData_Trim(data, alpha)
    print(ds.alpha)
    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_trim_fn,
        shuffle=shuffle
    )
    return dl