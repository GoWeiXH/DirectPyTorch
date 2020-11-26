"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: activation.py
@time: 2020/11/25 15:06
@description: 
"""

import torch.nn as nn


class Activation(nn.Module):

    def __init__(self, func_name, param=None, **options):
        super(Activation, self).__init__()
        self.func_name = func_name
        self.param = param
        self.activation = self.get_func(**options)

    def get_func(self, **options):
        if self.func_name == 'relu':
            func = nn.ReLU(**options)
        elif self.func_name == 'lrelu':
            func = nn.LeakyReLU(self.param) if self.param else nn.LeakyReLU()
        elif self.func_name == 'sigmoid':
            func = nn.Sigmoid()
        elif self.func_name == 'softmax':
            func = nn.Softmax(self.param) if self.param else nn.Softmax()
        else:
            func = None
        return func

    def forward(self, input_tensor):
        return self.activation(input_tensor)

    def __bool__(self):
        return bool(self.activation)
