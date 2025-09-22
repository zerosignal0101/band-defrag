import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    '''
    封装的统一网络初始化函数
    '''
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    '''
    把numpy转成torch.Tensor
    '''
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output
