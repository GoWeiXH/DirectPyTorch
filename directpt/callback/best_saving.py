"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: best_saving.py
@time: 2020/11/26 19:21
@description: 
"""

from typing import Union

import torch
from .backens import _worst


class BestSaving:
    """
    保存最优模型

    Args:
        save_path: 模型保存路径
        monitor: 保存模型的监测项目，支持 loss, val_loss, acc, val_acc
        check_freq: 检查频率,
                    值'epoch'表示频率为每个 epoch,
                    值为 int 表示 epoch 数目
        verbose: 是否显示详细信息

    """

    def __init__(self, save_path: str, monitor: str = 'val_loss',
                 verbose: int = 0, check_freq: Union[str, int] = 'epoch'):
        self.save_path = save_path
        self.monitor = monitor
        self.verbose = verbose
        self.best = _worst(self.monitor)
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
            'model_fit_logs': logs
        }, self.save_path)

        if self.verbose:
            print('Best model is saved to {}'.format(self.save_path))

    def best_save(self, model, logs):

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
                raise ValueError('Monitor must be selected from [acc, val_acc, loss, val_loss]')
        self._epoch_since_last_saving += 1

    def __str__(self):
        return 'best_saving'
