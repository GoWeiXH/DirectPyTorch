import time

import torch
import torch.nn as nn
from torch.nn import Module
from torch.tensor import Tensor
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from ..function import fill_nan
from ..utils import LogPrinter
from .._exception import MetricsError
from .step_metrics import StepMetrics


class Trainer:
    """
    训练器

    Args:
        model: 模型
        optimizer: 优化器
        loss: 损失函数
        main_device: 运行主设备

    """

    def __init__(self, model: Module, optimizer: Optimizer, loss: nn.Module,
                 main_device: str, threshold: float = 0.5):
        self.model = model
        self.main_device = main_device
        self.loss_func = loss
        self.optimizer = optimizer
        self.threshold = threshold
        self.metrics = list()
        self.logs = {'loss': []}
        self.multi = False

        self.__step_metrics = None
        self.__metric_func = dict()
        self.__metric_val_func = dict()

    def train_loss_step(self, batch_x: Tensor, batch_y: Tensor) -> Tensor:
        output = self.model(batch_x)
        step_loss = self.loss_func(output, batch_y)

        self.optimizer.zero_grad()
        step_loss.backward()
        self.optimizer.step()

        return step_loss

    def test_loss_step(self, y_pre: Tensor, batch_y: Tensor) -> Tensor:
        step_loss = self.loss_func(y_pre, batch_y)
        return step_loss

    def set_metrics(self, metrics, metric_func_lib, callbacks):
        # 验证评价方法是否合法
        metrics = [] if not metrics else metrics
        for m in metrics:
            if m not in metric_func_lib:
                raise MetricsError(m, list(metric_func_lib.keys()))

        # 将回调函数使用的 监控方法(monitor) 添加至 评价方法(metrics)
        for k, v in callbacks.items():
            monitor = v.monitor
            if monitor not in metrics:
                metrics.append(monitor)
        self.metrics = metrics

        # 获取 评价方法metrics 中的方法名称对应的计算方法
        for m in self.metrics:
            if 'val' in m:
                self.__metric_val_func[m] = metric_func_lib[m]
            else:
                self.__metric_func[m] = metric_func_lib[m]
        if 'loss' in self.__metric_func:
            del self.__metric_func['loss']

    def test(self, test_loader):

        correct_num, test_data_len = 0, 0
        steps_loss_sum, steps_num = 0, 0
        metrics_res = {item: [0, 0] for item in self.__metric_val_func.keys()}

        for step, (batch_x, batch_y) in enumerate(test_loader, start=1):
            steps_num = step
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)
            y_pre = self.model(batch_x)

            if 'val_loss' in self.__metric_val_func:
                test_loss = self.test_loss_step(y_pre, batch_y)
                steps_loss_sum += test_loss

            if 'val_acc' in self.__metric_val_func:
                correct, data_len = self.__step_metrics.test_acc_step(y_pre, batch_y)
                correct_num += correct
                test_data_len += data_len

            for item, func in self.__metric_val_func.items():
                if item not in ['val_loss', 'val_acc']:
                    mol, den = func(y_pre, batch_y)
                    metrics_res[item][0] += mol
                    metrics_res[item][1] += den

        if 'val_loss' in self.__metric_val_func:
            val_loss = steps_loss_sum / steps_num
            self.logs['val_loss'].append(val_loss.item())

        if 'val_acc' in self.__metric_val_func:
            val_acc = correct_num / test_data_len
            self.logs['val_acc'].append(val_acc.item())

        for item, res in metrics_res.items():
            if item not in ['val_loss', 'val_acc']:
                res = res[0] / res[1]
                val_res = fill_nan(res, 0)
                if self.multi:
                    val_res = torch.mean(val_res)
                self.logs[item].append(val_res.item())

    def train(self, train_loader: DataLoader, test_loader: DataLoader,
              metrics: list = None, epochs: int = 1, multi: bool = False,
              val_freq=1, callbacks: list = None):

        # 初始化日志打印类
        printer = LogPrinter(epochs, len(train_loader), val_freq)

        # 初始化 metrics 计算类
        self.multi = multi
        self.__step_metrics = StepMetrics(multi, self.threshold, self.main_device)
        metric_func_lib = self.__step_metrics.metric_func_lib

        # 根据指定参数设置回调函数
        callbacks = {str(cs): cs for cs in callbacks} if callbacks else {}
        # 设置 metrics
        self.set_metrics(metrics, metric_func_lib, callbacks)
        for item in self.metrics:
            self.logs[item] = []

        # 开始训练
        for e in range(1, epochs + 1):
            epoch_start = time.time()
            printer.epoch_start_log(e)

            train_step_logs = {item: [] for item in self.__metric_func.keys()}
            train_step_logs['loss'] = []
            for step, (batch_x, batch_y) in enumerate(train_loader, start=1):
                batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

                # 训练 loss
                step_loss = self.train_loss_step(batch_x, batch_y)
                train_step_logs['loss'].append(step_loss.item())

                y_pre = self.model(batch_x)

                for metrics, func in self.__metric_func.items():
                    res = func(y_pre, batch_y)
                    train_step_logs[metrics].append(res.item())

                # 打印 step 训练日志
                printer.step_train_log(step, train_step_logs)

            for item, his in train_step_logs.items():
                epoch_res = sum(his) / len(his)
                self.logs[item].append(epoch_res)

            # 打印 epoch 训练日志
            val_flag = e % val_freq == 0
            printer.epoch_end_log(epoch_start, self.logs, val_flag)

            # 计算验证集结果
            if val_flag:
                self.test(test_loader)
                printer.add_val_log(self.logs)

            if callbacks:
                if 'best_saving' in callbacks:
                    callbacks['best_saving'].best_save(self.model, self.logs)
                if 'early_stopping' in callbacks:
                    if callbacks['early_stopping'].early_stop(self.logs):
                        break

        return self.logs
