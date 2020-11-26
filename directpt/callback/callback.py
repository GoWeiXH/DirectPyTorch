"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: check.py
@time: 2020/11/26 13:52
@description: 
"""

import torch

MAX = torch.tensor(1e1000, dtype=torch.float64)
MIN = torch.tensor(-1e1000, dtype=torch.float64)


def _best(monitor):
    if monitor in ['loss', 'val_loss']:
        best = MAX
    elif monitor in ['acc', 'val_acc']:
        best = MIN
    else:
        msg = "Arg 'monitor' is invalid, choose from ['loss', 'val_loss', 'acc', 'val_acc'] "
        raise ValueError(msg)
    return best


class BestSaving:

    def __init__(self, save_path, monitor='val_loss', verbose=0, check_freq='epoch'):
        self.save_path = save_path
        self.monitor = monitor
        self.verbose = verbose
        self.best = _best(self.monitor)
        self._epoch_since_last_saving = 0

        if check_freq == 'epoch':
            self.check_freq = 1
        elif isinstance(check_freq, int):
            self.check_freq = check_freq
        else:
            ...

    def _save_model(self, model, logs):
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_logs': logs
        }, self.save_path)

    def save_best(self, model, logs):

        # todo freq
        # todo verbose

        cur_monitor_value = logs[self.monitor][-1]

        if self._epoch_since_last_saving % self.check_freq == 0:
            if 'loss' in self.monitor:
                if cur_monitor_value < self.best:
                    self.best = cur_monitor_value
                    self._save_model(model, logs)
            elif 'acc' in self.monitor:
                if cur_monitor_value > self.best:
                    self.best = cur_monitor_value
                    self._save_model(model, logs)
            else:
                ...
        self._epoch_since_last_saving += 1

    def __str__(self):
        return 'best_saving'


class EarlyStopping:

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.best = _best(self.monitor)
        self.verbose = verbose

    def early_stop(self, logs):

        # todo verbose

        cur_monitor_value = logs[self.monitor][-1]
        delta = cur_monitor_value - self.best

        if 'loss' in self.monitor:
            if delta < 0 and -delta > self.min_delta:
                self.best = cur_monitor_value
                self.wait = 0
            else:
                self.wait += 1
        elif 'acc' in self.monitor:
            if delta > 0 and delta > self.min_delta:
                self.best = cur_monitor_value
                self.wait = 0
            else:
                self.wait += 1
        else:
            ...

        return self.wait > self.patience

    def __str__(self):
        return 'early_stopping'
