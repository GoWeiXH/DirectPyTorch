import time

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from ..utils import LogPrinter
from ..metrics import binary_correct, multi_class_correct
from .._exception import MetricsError


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

        self.correct = None
        self.metric_func_lib = {
            'loss': self.train_loss_step,
            'val_loss': self.test_loss_step,
            'acc': self.train_acc_step,
            'val_acc': self.test_acc_step
        }

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

    def train_acc_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)
        step_correct = self.correct(self.model(batch_x), batch_y, threshold=self.threshold)
        step_acc = step_correct / len(batch_y)
        return step_acc

    def test_acc_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        step_acc = self.correct(self.model(batch_x), batch_y, threshold=self.threshold)
        return step_acc

    def train(self, train_loader: DataLoader, test_loader: DataLoader,
              metrics: list = None, epochs: int = 1, multi: bool = False,
              val_freq=1, callbacks: list = None):

        # 初始化日志打印类
        printer = LogPrinter(epochs, len(train_loader), val_freq)

        # 识别 多分类 / 二分类，对应不同计算准确率方法
        self.correct = multi_class_correct if multi else binary_correct

        # 验证评价方法是否合法
        metrics = [] if not metrics else metrics
        for m in metrics:
            if m not in self.metric_func_lib.keys():
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

        # 获取 评价方法metrics 中的方法名称对应的计算方法
        metric_func, metric_val_func = {}, {}
        for m in metrics:
            if 'val' in m:
                metric_val_func[m] = self.metric_func_lib[m]
            else:
                metric_func[m] = self.metric_func_lib[m]

        # 开始训练
        logs = {item: [] for item in self.metric_func_lib.keys()}
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
                if 'acc' in metric_func:
                    step_acc = self.train_acc_step(batch_x, batch_y)
                    step_acc_his.append(step_acc)

                # 打印 step 训练日志
                printer.step_log(step, step_loss, step_acc)

            epoch_loss = sum(step_loss_his) / len(step_loss_his)
            logs['loss'].append(epoch_loss)

            epoch_acc = None
            if 'acc' in metric_func:
                epoch_acc = sum(step_acc_his) / len(step_acc_his)
                logs['acc'].append(epoch_acc)

            # 打印 epoch 训练日志
            val_flag = e % val_freq == 0
            printer.epoch_end_log(epoch_start, epoch_loss, epoch_acc, val_flag)

            # 计算验证集结果
            if val_flag:
                correct_num, test_data_len = 0, 0
                steps_loss_list = []
                for step, (batch_x, batch_y) in enumerate(test_loader, start=1):
                    batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

                    if 'val_acc' in metric_val_func:
                        correct_num += self.test_acc_step(batch_x, batch_y)
                        test_data_len += len(batch_y)

                    if 'val_loss' in metric_val_func:
                        steps_loss_list.append(self.test_loss_step(batch_x, batch_y))

                val_acc = correct_num / test_data_len
                if steps_loss_list:
                    val_loss = torch.mean(torch.tensor(steps_loss_list))
                    logs['val_loss'].append(val_loss)

                logs['val_acc'].append(val_acc)
                printer.add_val_log(logs)

            if callbacks:
                if 'best_saving' in callbacks:
                    callbacks['best_saving'].best_save(self.model, logs)
                elif 'early_stopping' in callbacks:
                    if callbacks['early_stopping'].early_stop(logs):
                        break

        return logs
