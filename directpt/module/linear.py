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
from torch.nn import Module
import torch.nn.functional as F

from .activation import Activation


class Linear(nn.Linear):
    """
    全连接层，可指定激活函数

    Args:
        in_features: 输入神经元个数
        out_features: 输出神经元个数
        bias: 是否使用偏置项
        activation: 激活函数名称
        activation_func: 激活函数，若指定此参数，则 activation 失效
        activation_param: 激活函数参数

    Examples:
        >>> layer = Linear(64, 32, activation='sigmoid')

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = None, activation_func: Module = None,
                 activation_param: float = None, **options) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

        self.activation = Activation(activation, activation_param, **options)
        if activation_func:
            self.activation = activation_func

    def forward(self, input_tensor: Tensor) -> Tensor:
        if bool(self.activation):
            return self.activation(F.linear(input_tensor, self.weight, self.bias))
        else:
            return F.linear(input_tensor, self.weight, self.bias)
