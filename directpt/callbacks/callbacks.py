from typing import Union

import torch

from .._exception import MonitorError
from ..function import worst


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

    Examples:

        >>> best_saving = BestSaving('model_save_path.pt', 'val_loss', verbose=1, check_freq=2)

    """

    def __init__(self, save_path: str, monitor: str = 'val_loss',
                 verbose: int = 0, check_freq: Union[str, int] = 'epoch'):
        self.save_path = save_path
        self.monitor = monitor
        self.verbose = verbose
        self.best = worst(self.monitor)
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

        if logs[self.monitor]:
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
                    raise MonitorError(self.monitor)
            self._epoch_since_last_saving += 1

    def __str__(self):
        return 'best_saving'


class EarlyStopping:
    """
    早停机制

    Args:
        monitor: 早停机制的监测项目，支持 loss, val_loss, acc, val_acc
        min_delta: 最小差值，小于此值时增加等待次数
        patience: 允许等待次数，大于此值退出训练
        verbose: 是否显示详细信息

    Examples:

        >>> early_stopping = EarlyStopping(monitor='val_acc', min_delta=1e-5, patience=50, verbose=1)

    """

    def __init__(self, monitor: str = 'val_loss', min_delta: float = 0, patience: int = 10, verbose: object = 0) -> object:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.best = worst(self.monitor)
        self.verbose = verbose

    def early_stop(self, logs):

        cur_monitor_value = logs[self.monitor][-1]
        delta = cur_monitor_value - self.best

        if 'loss' in self.monitor:
            if delta < 0 and -delta > self.min_delta:
                self.best = cur_monitor_value
                self.wait = 0
            else:
                self.wait += 1
                if self.verbose:
                    print('No improvement to best {}: {:.4} - [{}]'.format(
                        self.monitor, self.best, self.wait))
        elif 'acc' in self.monitor:
            if delta > 0 and delta > self.min_delta:
                self.best = cur_monitor_value
                self.wait = 0
            else:
                self.wait += 1
                if self.verbose:
                    print('No improvement to best {}: {:.4} - [{}]'.format(
                        self.monitor, self.best, self.wait))
        else:
            raise MonitorError(self.monitor)

        if self.verbose:
            print('Early Stop - best - {}: {}'.format(self.monitor, self.best))

        return self.wait > self.patience

    def __str__(self):
        return 'early_stopping'
