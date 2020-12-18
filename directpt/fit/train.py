import time

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from ..utils import Printer
from ..module.metrics import Correct
from ..exception import MetricsError


class Trainer:
    """
    训练器

    Args:
        model: 模型
        optimizer: 优化器
        loss: 损失函数
        device: 运行硬件设备

    """

    def __init__(self, model: Module, optimizer: Optimizer, loss: nn.Module,
                 pre_threshold: float = 0.5, device: str = 'cpu'):
        self.model = model
        self.loss_func = loss
        self.optimizer = optimizer
        self.acc_func = Correct(threshold=pre_threshold)

        self.main_device, self.device, self.is_parallel = self.set_train_device(device)

        self.metric_func_lib = {
            'loss': self.train_loss_step,
            'val_loss': self.test_loss_steps,
            'acc': self.train_acc_step,
            'val_acc': self.test_acc_steps
        }

    def set_train_device(self, device: str) -> (str, str, bool):
        if device == 'gpu':
            if torch.cuda.is_available():
                main_device = 'cuda:0'
            else:
                print('GPU is not available, CPU is used by default')
                main_device = 'cpu'
        elif device.startswith('gpu'):
            main_device = device
        elif isinstance(device, list):
            main_device = 'cuda:{}'.format(device[0])
        else:
            main_device = 'cpu'

        is_parallel = False
        if isinstance(device, list):
            if len(device) > 1 and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model, device_ids=device)
                is_parallel = True

        self.model.to(main_device)

        device = 'cpu' if main_device == 'cpu' else 'gpu'
        return main_device, device, is_parallel

    def train_loss_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        output = self.model(batch_x)
        step_loss = self.loss_func(output, batch_y)

        self.optimizer.zero_grad()
        step_loss.backward()
        self.optimizer.step()

        return step_loss.item()

    def train_acc_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)
        step_correct = self.acc_func(self.model(batch_x), batch_y)
        step_acc = step_correct / len(batch_y)
        return step_acc

    def test_loss_steps(self, data_loader: DataLoader):
        step_loss_list = []
        for step, (batch_x, batch_y) in enumerate(data_loader, start=1):
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

            output = self.model(batch_x)
            step_loss = self.loss_func(output, batch_y)
            step_loss_list.append(step_loss.item())
        mean_step_loss = torch.mean(torch.tensor(step_loss_list))
        return mean_step_loss.item()

    def test_acc_steps(self, data_loader: DataLoader):
        sample_num, correct_num = 0, 0
        for step, (batch_x, batch_y) in enumerate(data_loader, start=1):
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)
            step_correct = self.acc_func(self.model(batch_x), batch_y)

            correct_num += step_correct
            sample_num += len(batch_y)

        mean_step_acc = correct_num / sample_num
        return mean_step_acc

    def train(self, train_loader: DataLoader, test_loader: DataLoader,
              metrics: list = None, epochs: int = 1,
              val_freq=1, callbacks: list = None):

        # 初始化日志打印类
        printer = Printer(epochs, len(train_loader), val_freq)

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

            if val_flag:
                for metric, val_func in metric_val_func.items():
                    metric_res = val_func(test_loader)
                    logs[metric].append(metric_res)

                printer.add_val_log(logs)

            if callbacks:
                if 'best_saving' in callbacks:
                    callbacks['best_saving'].best_save(self.model, logs)
                elif 'early_stopping' in callbacks:
                    if callbacks['early_stopping'].early_stop(logs):
                        break

        return logs
