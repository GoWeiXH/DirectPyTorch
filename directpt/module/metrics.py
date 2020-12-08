import torch
import torch.nn as nn

from ..functional import correct, mc_correct


def accuracy(y_pre: torch.Tensor, y_true: torch.Tensor):
    """
    计算二分类测正确样本的数量

    Args:
        y_pre: 预测结果
        y_true: 真实标签结果
    """

    return correct(y_pre, y_true) / len(y_true)


def mc_accuracy(y_pre: torch.Tensor, y_true: torch.Tensor):
    """
    计算多分类测正确样本的数量

    Args:
        y_pre: 预测结果
        y_true: 真实标签结果
    """

    y_true = torch.argmax(y_true, dim=1)
    y_pre = torch.argmax(y_pre, dim=1)
    return mc_correct(y_pre, y_true) / len(y_true)


def recall(y_pre: torch.Tensor, y_true: torch.Tensor):
    """
    计算分类召回值 (Recall)

    Args:
        y_pre: 预测结果
        y_true: 真实标签结果
    """
    # todo


def precision(y_pre: torch.Tensor, y_true: torch.Tensor):
    """
    计算分类精确度 (precision)
    Args:
        y_pre: 预测结果
        y_true: 真实标签结果
    """
    # todo


def f_beta_score(y_pre: torch.Tensor, y_true: torch.Tensor):
    """
    计算分类 F-Beta 值

    Args:
        y_pre: 预测结果
        y_true: 真实标签结果
    """
    # todo


def f1_score(y_pre: torch.Tensor, y_true: torch.Tensor):
    """
    计算分类 F1 值

    Args:
        y_pre: 预测结果
        y_true: 真实标签结果
    """
    # todo


class MacroCostLoss(nn.Module):

    """
    F1-Score 损失函数

    Args:
        label_weight: 类别标签权重
        sample_weight: 样本标签权重

    """

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
    多标签分类交叉熵
    Multi-Label Category Cross Entropy

    Args:
        label_weight: 类别标签权重
        sample_weight: 样本标签权重
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
