"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: linear.py
@time: 2020/11/25 13:49
@description: 
"""

from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from .activation import Activation


class Linear(nn.Linear):
    """
    全连接层，可指定激活函数、激活函数的参数

    Args:
        in_features: 输入神经元个数
        out_features: 输出神经元个数
        bias: 是否使用偏置项
        activation: 激活函数名称, 字符串指定
        activation_func: pytorch内置激活函数，若指定此参数，则 activation 失效
        activation_param: 激活函数参数，单参数传入或字典传入

    Examples:
        >>> x = torch.randn(10, 32, dtype=torch.float)
        >>> layer = Linear(32, 16, activation='relu')
        >>> y = layer(x)

        or

        >>> l_relu = torch.nn.LeakyReLU()
        >>> layer = Linear(32, 16, activation_func=l_relu, activation_param=0.2)

        >>> layer = Linear(32, 16, activation_func=l_relu, activation_param={'negative_slope': 0.2})
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 activation: str = None,
                 activation_func: Module = None,
                 activation_param: Union[float, dict] = None,
                 **options) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

        self.activation = Activation(activation, activation_param, **options)
        if activation_func:
            self.activation = activation_func

    def forward(self, input_tensor: Tensor) -> Tensor:
        if bool(self.activation):
            return self.activation(F.linear(input_tensor, self.weight, self.bias))
        else:
            return F.linear(input_tensor, self.weight, self.bias)
