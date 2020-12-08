from .backens import _worst


class EarlyStopping:

    """
    早停机制

    Args:
        monitor: 早停机制的监测项目，支持 loss, val_loss, acc, val_acc
        min_delta: 最小差值，小于此值时增加等待次数
        patience: 允许等待次数，大于此值退出训练
        verbose: 是否显示详细信息

    """

    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.best = _worst(self.monitor)
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
