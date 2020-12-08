import torch

MAX = torch.tensor(1e1000, dtype=torch.float64)
MIN = torch.tensor(-1e1000, dtype=torch.float64)


def _worst(monitor):
    if monitor in ['loss', 'val_loss']:
        best = MAX
    elif monitor in ['acc', 'val_acc']:
        best = MIN
    else:
        msg = "Arg 'monitor' is invalid, choose from ['loss', 'val_loss', 'acc', 'val_acc'] "
        raise ValueError(msg)
    return best
