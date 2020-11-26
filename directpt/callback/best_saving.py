"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: best_saving.py
@time: 2020/11/26 19:21
@description: 
"""

import torch
from .backens import _best


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
