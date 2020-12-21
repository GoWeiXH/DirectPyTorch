import torch

from .exception import MonitorError


MAX = torch.tensor(1e1000, dtype=torch.float64)
MIN = torch.tensor(-1e1000, dtype=torch.float64)


def worst(monitor):
    if monitor in ['loss', 'val_loss']:
        best = MAX
    elif monitor in ['acc', 'val_acc']:
        best = MIN
    else:
        raise MonitorError(monitor)
    return best
