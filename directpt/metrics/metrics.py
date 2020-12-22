import torch
import torch.nn as nn

from ..backend import confusion_matrix, mc_confusion_matrix


class Correct(nn.Module):

    def __init__(self, threshold=0.5):
        super(Correct, self).__init__()

        self.threshold = threshold

    def forward(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        y_pre = torch.as_tensor(y_pre > self.threshold, dtype=torch.int)
        return torch.as_tensor((y_pre == y_true), dtype=torch.int).sum().item()


class Accuracy(nn.Module):
    """
    计算 二/多分类 准确率
    """

    def __init__(self, threshold=0.5, multi_class=False):
        super(Accuracy, self).__init__()

        self.threshold = threshold

        self.correct = Correct()

        if multi_class:
            self.accuracy_method = self.multi_acc
        else:
            self.accuracy_method = self.binary_acc

    def forward(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        return self.accuracy_method(y_pre, y_true)

    def binary_acc(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        correct = self.correct(y_pre, y_true)
        return correct / len(y_pre)

    def multi_acc(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        y_pre, y_true = torch.argmax(y_pre, dim=1), torch.argmax(y_true, dim=1)
        correct = self.correct(y_pre, y_true)
        return correct / len(y_pre)


class Recall(nn.Module):

    def __init__(self, threshold=0.5, multi_class=False):
        super(Recall, self).__init__()

        self.threshold = threshold

        if multi_class:
            self.recall_method = self.multi_recall
        else:
            self.recall_method = self.binary_recall

    def forward(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        return self.recall_method(y_pre, y_true)

    def binary_recall(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        y_pre = torch.as_tensor(y_pre > self.threshold, dtype=torch.int)
        tp, fn, fp, tn = confusion_matrix(y_pre, y_true)
        return tp / (tp + fn)

    def multi_recall(self, y_pre: torch.Tensor, y_true: torch.Tensor):
        ...


class Precision(nn.Module):
    ...


class FBetaScore(nn.Module):
    ...


class F1Score(nn.Module):
    ...


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
