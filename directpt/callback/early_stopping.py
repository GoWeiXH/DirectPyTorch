"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: earlystop.py
@time: 2020/11/26 19:21
@description: 
"""

import torch
from .backens import _best


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
