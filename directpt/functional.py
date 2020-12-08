import torch


def correct(y_pre: torch.Tensor, y_true: torch.Tensor, threshold=0.5):
    """
    计算二分类预测正确样本的数量

    Args:
        y_pre: 预测结果
        y_true: 真实标签结果
        threshold: 判断类别的阈值
    """
    y_pre = torch.as_tensor(y_pre > threshold, dtype=torch.int)
    return torch.sum(torch.as_tensor(y_pre == y_true, dtype=torch.int)).item()


def mc_correct(y_pre: torch.Tensor, y_true: torch.Tensor):
    """
    计算多分类预测正确样本的数量

    Args:
        y_pre: 预测结果
        y_true: 真实标签结果
    """
    return torch.tensor(torch.argmax(y_pre, dim=1) == torch.argmax(y_true, dim=1)).sum()
