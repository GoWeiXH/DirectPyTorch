"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: metrics.py
@time: 2020/11/27 17:54
@description: 
"""

import torch
import torch.nn as nn

from ..functional import correct, mc_correct


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        return correct(y_pre, y_true) / len(y_true)


class MCAccuracy(nn.Module):
    def __init__(self):
        super(MCAccuracy, self).__init__()

    def forward(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        y_true = torch.argmax(y_true, dim=1)
        y_pre = torch.argmax(y_pre, dim=1)
        return mc_correct(y_pre, y_true) / len(y_true)


class MacroCostLoss(nn.Module):

    def __init__(self, label_weight=None, sample_weight=None):
        super(MacroCostLoss, self).__init__()
        self.label_weight = label_weight
        self.sample_weight = sample_weight

    def forward(self, y_pre, y_true):
        # todo label_weight, sample_weight

        tp = torch.sum(y_pre * y_true * self.label_weight, dim=1)
        fp = torch.sum((1 - y_pre) * y_true, dim=1)
        fn = torch.sum(y_pre * (1 - y_true), dim=1)
        soft_f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
        cost = 1 - soft_f1
        return torch.mean(cost)


class MultiLabelCCE(nn.Module):
    """
    MultiLabelCategoricalCrossEntropy
    """

    def __init__(self, label_weight=None, sample_weight=None):
        super(MultiLabelCCE, self).__init__()
        self.label_weight = label_weight
        self.sample_weight = sample_weight

    def forward(self, y_pre, y_true):
        # todo label_weight, sample_weight

        y_pre = (1 - 2 * y_true) * y_pre
        y_pre_neg = y_pre - y_true * 1e12
        y_pre_pos = y_pre - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pre[..., :1])
        y_pre_neg = torch.cat([y_pre_neg, zeros], dim=-1)
        y_pre_pos = torch.cat([y_pre_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pre_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pre_pos, dim=-1)
        mlcce_loss = neg_loss + pos_loss
        return torch.mean(mlcce_loss)
