from torch.nn.modules.loss import _Loss

import math


import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 123 # or whatever you choose
# random.seed(SEED) # if you're using random
# np.random.seed(123) # if you're using numpy
torch.manual_seed(SEED) # torch.cuda.manual_seed_all(SEED) is not required
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FairseqCriterion(_Loss):

    def __init__(self, args, task):
        super().__init__()
        self.args = args
        self.padding_idx = task.target_dictionary.pad()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        raise NotImplementedError

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError

    @staticmethod
    def grad_denom(sample_sizes):
        """Compute the gradient denominator for a set of sample sizes."""
        return sum(sample_sizes)



class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    def forward(self, output, target):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        output = output.cpu()
        target = target.cpu()
        # print("output: ", torch.sum(output))
        #print("target: ", torch.sum(target))
        lprobs = F.log_softmax(output, dim=-1, dtype=torch.float32)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        # print("nll_loss: ", torch.sum(nll_loss))
        #print("smooth_loss: ", torch.sum(smooth_loss))
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        return loss