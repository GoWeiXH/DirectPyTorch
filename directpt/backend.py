import torch

from ._exception import MonitorError

MAX = torch.tensor(1e1000, dtype=torch.float64)
MIN = torch.tensor(-1e1000, dtype=torch.float64)


def worst(monitor):
    if monitor in ['loss', 'val_loss']:
        best = MAX
    elif monitor in ['acc', 'val_acc']:
        best = MIN
    else:
        raise MonitorError(monitor, ['loss', 'val_loss', 'acc', 'val_acc', 'val_recall'])
    return best


def fill_nan(x, value):
    return torch.where(torch.isnan(x), torch.full_like(x, value), x)


