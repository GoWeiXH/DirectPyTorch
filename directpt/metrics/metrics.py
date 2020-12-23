import torch
import torch.nn as nn

from ..backend import fill_nan


def confusion_matrix(y_pre: torch.Tensor, y_true: torch.Tensor,
                     threshold, multi):
    y_pre = torch.as_tensor(y_pre > threshold, dtype=torch.int)
    dim = 1 if multi else 0
    tp = torch.sum(y_pre * y_true, dim=dim)
    fn = torch.sum((1 - y_pre) * y_true, dim=dim)
    fp = torch.sum(y_pre * (1 - y_true), dim=dim)
    tn = torch.sum((1 - y_pre) * (1 - y_true), dim=dim)
    return tp, fn, fp, tn


def binary_correct(y_pre: torch.Tensor, y_true: torch.Tensor, threshold=0.5):
    y_pre = torch.as_tensor(y_pre > threshold, dtype=torch.int)
    same = torch.as_tensor(y_pre == y_true, dtype=torch.int)
    return torch.sum(same)


def multi_class_correct(y_pre: torch.Tensor, y_true: torch.Tensor, threshold=0.5):
    y_pre, y_true = y_pre.argmax(dim=1), y_true.argmax(dim=1)
    same = torch.as_tensor(y_pre == y_true, dtype=torch.int)
    return torch.sum(same)


def binary_accuracy(y_pre: torch.Tensor, y_true: torch.Tensor, threshold=0.5):
    tp, fn, fp, tn = confusion_matrix(y_pre, y_true, threshold, multi=False)
    return (tp + tn) / (tp + fn + fp + tn)


def multi_class_accuracy(y_pre: torch.Tensor, y_true: torch.Tensor):
    y_pre, y_true = torch.argmax(y_pre, dim=1), torch.argmax(y_true, dim=1)
    res = multi_class_correct(y_pre, y_true)
    return res / len(y_pre)


def recall_precision_fscore(y_pre: torch.Tensor, y_true: torch.Tensor,
                            multi=False,
                            threshold=0.5, beta=1.0, zero_division=0):
    if multi:
        y_pre, y_true = y_pre.T, y_true.T

    tp, fn, fp, _ = confusion_matrix(y_pre, y_true, threshold, multi)

    recall = fill_nan(tp / (tp + fn), zero_division)
    precision = fill_nan(tp / (tp + fp), zero_division)

    beta2 = beta ** 2
    f_score = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    f_score = fill_nan(f_score, zero_division)

    if multi:
        recall = torch.mean(recall)
        precision = torch.mean(precision)
        f_score = torch.mean(f_score)

    return recall, precision, f_score


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
