import torch


def confusion_matrix(y_pre: torch.Tensor, y_true: torch.Tensor, threshold=0.5):
    y_pre = torch.as_tensor(y_pre > threshold, dtype=torch.int)

    tp = torch.sum(y_pre * y_true)
    fn = torch.sum((1 - y_pre) * y_true)
    fp = torch.sum(y_pre * (1 - y_true))
    tn = torch.sum((1 - y_pre) * (1 - y_true))

    return tp, fn, fp, tn


def mc_confusion_matrix(y_pre: torch.Tensor, y_true: torch.Tensor, threshold=0.5, multi_class=False):
    # todo
    ...
