"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: linear.py
@time: 2020/11/25 13:49
@description: 
"""

import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .activation import Activation


class Linear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = None, activation_param: float = None) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

        self.activation = Activation(activation, activation_param)

    def forward(self, input_tensor: Tensor) -> Tensor:
        if bool(self.activation):
            return self.activation(F.linear(input_tensor, self.weight, self.bias))
        else:
            return F.linear(input_tensor, self.weight, self.bias)
