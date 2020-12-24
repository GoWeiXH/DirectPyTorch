import time

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

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

    def train_loss_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        output = self.model(batch_x)
        step_loss = self.loss_func(output, batch_y)

        self.optimizer.zero_grad()
        step_loss.backward()
        self.optimizer.step()

        return step_loss.item()

    def test_loss_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        output = self.model(batch_x)
        step_loss = self.loss_func(output, batch_y)
        return step_loss.item()

    def test(self, test_loader):

        correct_num, test_data_len = 0, 0
        steps_loss_list = []
        for step, (batch_x, batch_y) in enumerate(test_loader, start=1):
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

            if 'val_loss' in self.__metric_val_func:
                steps_loss_list.append(self.test_loss_step(batch_x, batch_y))

            if 'val_acc' in self.__metric_val_func:
                correct_num += self.__step_metrics.test_acc_step(self.model, batch_x, batch_y)
                test_data_len += len(batch_y)

        if 'val_loss' in self.__metric_val_func:
            val_loss = torch.mean(torch.tensor(steps_loss_list))
            self.logs['val_loss'].append(val_loss)

        if 'val_acc' in self.__metric_val_func:
            val_acc = correct_num / test_data_len
            self.logs['val_acc'].append(val_acc)

    def train(self, train_loader: DataLoader, test_loader: DataLoader,
              metrics: list = None, epochs: int = 1, multi: bool = False,
              val_freq=1, callbacks: list = None):

        # 初始化日志打印类
        printer = LogPrinter(epochs, len(train_loader), val_freq)

        # 初始化 metrics 计算类
        self.__step_metrics = StepMetrics(multi, self.threshold)
        metric_func_lib = self.__step_metrics.metric_func_lib

        # 验证评价方法是否合法
        metrics = [] if not metrics else metrics
        for m in metrics:
            if m not in metric_func_lib:
                raise MetricsError(m)

        # 根据指定参数设置回调函数
        callbacks = {str(cs): cs for cs in callbacks} if callbacks else {}

        # 将回调函数使用的 监控方法(monitor) 添加至 评价方法(metrics)
        callbacks_func = {}
        for k, v in callbacks.items():
            callbacks_func[k] = v
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

        # 开始训练
        for item in self.metrics:
            self.logs[item] = []
        for e in range(1, epochs + 1):
            epoch_start = time.time()
            printer.epoch_start_log(e)

            step_loss_his, step_acc_his = [], []
            for step, (batch_x, batch_y) in enumerate(train_loader, start=1):
                batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

                # 训练 loss
                step_loss = self.train_loss_step(batch_x, batch_y)
                step_loss_his.append(step_loss)

                # 训练 acc
                step_acc = None
                if 'acc' in self.__metric_func:
                    step_acc = self.__step_metrics.train_acc_step(self.model, batch_x, batch_y)
                    step_acc_his.append(step_acc)

                # 打印 step 训练日志
                printer.step_log(step, step_loss, step_acc)

            epoch_loss = sum(step_loss_his) / len(step_loss_his)
            self.logs['loss'].append(epoch_loss)

            epoch_acc = None
            if 'acc' in self.__metric_func:
                epoch_acc = sum(step_acc_his) / len(step_acc_his)
                self.logs['acc'].append(epoch_acc)

            # 打印 epoch 训练日志
            val_flag = e % val_freq == 0
            printer.epoch_end_log(epoch_start, epoch_loss, epoch_acc, val_flag)

            # 计算验证集结果
            if val_flag:
                self.test(test_loader)
                printer.add_val_log(self.logs)

            if callbacks:
                if 'best_saving' in callbacks:
                    callbacks['best_saving'].best_save(self.model, self.logs)
                elif 'early_stopping' in callbacks:
                    if callbacks['early_stopping'].early_stop(self.logs):
                        break

        return self.logs
