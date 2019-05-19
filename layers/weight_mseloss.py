# -*- coding: utf-8 -*-
# Original author: yq_yao


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class WeightMseLoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightMseLoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, weights):
        ''' inputs is N * C
            targets is N * C
            weights is N * C
        '''
        N = inputs.size(0)
        C = inputs.size(1)

        out = (targets - inputs)
        out = weights * torch.pow(out, 2)
        loss = out.sum()

        if self.size_average:
            loss = loss / (N * C)
        return loss
class FsvmLoss(nn.Module):
    def __init__(self, size_average=True):
        super(FsvmLoss, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, bound = 0.1, upper = 0.1):
        ''' inputs is N * C
            targets is N * C
            weights is N * C
        '''
        
        N = inputs.size(0)
        C = inputs.size(1)

        out = (targets - inputs)
        #out = weights * torch.pow(out, 2)
        
        out = nn.functional.relu(out - upper ) + nn.functional.relu( -out - bound )
        loss = out.sum()

        if self.size_average:
            loss = loss / (N * C)
        return loss
class FsvmLoss_norm(nn.Module):
    def __init__(self, size_average=True):
        super(FsvmLoss_norm, self).__init__()
        self.size_average = size_average

    def forward(self, inputs, targets, scale, weights , bound = 0.1):
        ''' inputs is N * C
            targets is N * C
            weights is N * C
        '''
        
        N = inputs.size(0)
        C = inputs.size(1)

        out = (targets - inputs)/(scale + 1e-6)
        #out = weights * torch.pow(out, 2)
        
        out = nn.functional.relu( torch.abs(out) - bound )*weights 
        loss = out.sum()

        if self.size_average:
            loss = loss / (N * C)
        return loss