import torch

from torch.distributions import Normal as TorchNormal

from myTransformer.MyUtils import clamp_preserve_gradients

### https://github.com/shchur/ifl-tpp/blob/master/code/dpp/distributions/normal.py ###
class Normal(TorchNormal):
    def log_cdf(self, x):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, x):
        # No numerically stable implementation of log survival is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(x), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)