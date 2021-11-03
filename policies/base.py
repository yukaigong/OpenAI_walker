import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sqrt

def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0,1)
        m.weight.data *= 1/torch.sqrt(m.weight.data.pow(2).sum(1,keepdim = True))
        if m.bias is not None:
            m.bias.data.fill_(0)
