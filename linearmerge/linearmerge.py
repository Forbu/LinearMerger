"""
In this module we will create the torch module that will optimize the linear layers of the network.
"""
import math

import torch
from torch import nn

class LinearMerger(nn.Module):
    """
    Break symetries in neural networks weights.
    """
    def __init__(self, in_dim, out_dim, bias=True, random_rows=True, permute_order=None):
        super(LinearMerger, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))


        # compute a triangular mask
        mask = (
            torch.ones(out_dim, in_dim)
        )

        # if we want to randomize the rows of the mask
        if random_rows and permute_order is None:
            mask[torch.randperm(out_dim).tolist(), torch.arange(in_dim)] = 0
        elif permute_order is not None:
            mask[permute_order.tolist(), torch.arange(in_dim)] = 0

        print(torch.sum(mask == 0))
        
        # we register the mask as a buffer so that it is moved to the device along with the module
        self.register_buffer("mask", mask)


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)

        if not hasattr(self, "bias"):
            return

        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # we apply the triangular mask to the weight matrix
        weight = self.weight * self.mask

        if self.bias is None:
            return nn.functional.linear(input, weight)
        else:
            return nn.functional.linear(input, weight, self.bias)