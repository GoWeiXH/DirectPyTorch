"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: functional.py
@time: 2020/12/7 17:36
@description: 
"""

import torch


def correct(y_pre: torch.Tensor, y_true: torch.Tensor, threshold=0.5):
    y_pre = torch.as_tensor(y_pre > threshold, dtype=torch.int)
    return torch.sum(torch.as_tensor(y_pre == y_true, dtype=torch.int)).item()


def mc_correct(y_pre: torch.Tensor, y_true: torch.Tensor):
    return torch.tensor(torch.argmax(y_pre, dim=1) == torch.argmax(y_true, dim=1)).sum()
