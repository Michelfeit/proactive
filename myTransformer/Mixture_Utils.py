import torch

from myTransformer.Utils import LabelSmoothingLoss
from myTransformer.MyUtils import clamp_preserve_gradients
from myTransformer.distributions import LogNormalMixtureDistribution
import myTransformer.Constants as Constants
from trim_process import EventData_Trim

### https://github.com/shchur/ifl-tpp/blob/master/code/dpp/models/log_norm_mix.py ###
def get_inter_time_dist(model, opt, event_data:EventData_Trim, raw_params) -> torch.distributions.Distribution:
    """
    Get the distribution over inter-event times given the context.
    Args:
        context: Context vector used to condition the distribution of each event,
            shape (batch_size, seq_len, context_size)
    Returns:
        dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)
    """
    # Slice the tensor to get the parameters of the mixture
    log_mean, log_std = event_data._get_statistics(opt)

    locs = raw_params[..., :model.num_mix_components]
    log_scales = raw_params[..., model.num_mix_components: (2 * model.num_mix_components)]
    log_weights = raw_params[..., (2 * model.num_mix_components):]

    log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
    log_weights = torch.log_softmax(log_weights, dim=-1)
    
    return LogNormalMixtureDistribution(
        locs=locs,
        log_scales=log_scales,
        log_weights=log_weights,
        mean_log_inter_time=log_mean,
        std_log_inter_time=log_std
    )

def log_probability(inter_time_dist:LogNormalMixtureDistribution, event_time_gap):
    batch_mask = get_non_pad_mask(event_time_gap)
    inter_times = event_time_gap.clamp(1e-10)

    log_p = inter_time_dist.log_prob(inter_times)  # (batch_size, seq_len)

    # Survival probability of the last interval (from t_N to t_end).
    # You can comment this section of the code out if you don't want to implement the log_survival_function 
    # for the distribution that you are using. This will make the likelihood computation slightly inaccurate,
    # but the difference shouldn't be significant if you are working with long sequences.
    # last_event_idx = get_non_pad_mask(inter_times).sum(-1, keepdim=True).long()  # (batch_size, 1) TODO
    # log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len)
    # log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,)
    # if self.num_marks > 1:
    #     mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)  # (batch_size, seq_len, num_marks)
    #     mark_dist = Categorical(logits=mark_logits)
    #     log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)
    log_p *= batch_mask  # (batch_size, seq_len)
    return log_p.sum(-1) #+ log_surv_last  # (batch_size,)

### taken form proactive ###
def type_loss(prediction, types, loss_func):
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]
    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num

def time_loss_gmm(samples, time_gap):
    diff = samples - time_gap[:,1:]
    se = torch.sum(torch.abs(diff))
    return se

def get_next_time_prediction(samples):
    preds = []
    j = 0
    for sequence in samples:
        i = 0
        for time in sequence[1:]:
            if(time == 0):
                break
            i += 1
        preds.append(samples[j][i])
        j+=1
    next_time = torch.tensor(preds)
    return next_time

# returns all positions that are not padding. the non pad positions are marked with a float 1.
def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    # only possible if dimension is 2
    assert seq.dim() == 2
    # unsqueeze adds a dimension at the specified index -> -1 adds the dimension at the end.
    mask = seq.ne(Constants.PAD).type(torch.float)
    mask[:, 0] = 1
    return mask