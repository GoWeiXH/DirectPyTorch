import time

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader

from .backens import print_train_log
from ..module.metrics import Correct


class Trainer:
    """
    训练器

    Args:
        model: 模型
        optimizer: 优化器
        acc: 准确度
        loss: 损失函数
        device: 运行硬件设备

    """

    def __init__(self, model: Module, optimizer: Optimizer,
                 loss: nn.Module, pre_threshold=0.5, device=None):
        self.model = model
        self.loss_func = loss
        self.optimizer = optimizer
        self.acc_func = Correct(threshold=pre_threshold)

        self.main_device, self.device, self.is_parallel = self.set_train_device(device)

        self.metric_func_lib = {
            'loss': self.train_loss_steps,
            'val_loss': self.test_loss_steps,
            'acc': self.acc_steps,
            'val_acc': self.acc_steps
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

    def acc_steps(self, data_loader: DataLoader):
        sample_num, correct_num = 0, 0
        for step, (batch_x, batch_y) in enumerate(data_loader, start=1):
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)
            step_correct = self.acc_func(self.model(batch_x), batch_y)

            correct_num += step_correct
            sample_num += len(batch_y)

        mean_step_acc = correct_num / sample_num
        return mean_step_acc

    def train_loss_steps(self, data_loader: DataLoader):
        total_step = len(data_loader)
        step_loss_list = []
        for step, (batch_x, batch_y) in enumerate(data_loader, start=1):
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

            output = self.model(batch_x)
            step_loss = self.loss_func(output, batch_y)
            step_loss_list.append(step_loss.item())

            self.optimizer.zero_grad()
            step_loss.backward()
            self.optimizer.step()

            past = int(step / total_step * 29)
            bar = '=' * past + '>' + '.' * (29 - past)
            pad_len = ' ' * (len(str(total_step)) - len(str(step))) + str(step)
            print('\r{}/{} [{}]'.format(pad_len, total_step, bar), end='', flush=True)

        total_step, bar = len(data_loader), '=' * 30
        print('\r{}/{} [{}]'.format(total_step, total_step, bar), end='', flush=True)
        mean_step_loss = torch.mean(torch.tensor(step_loss_list)).item()
        return mean_step_loss

    def test_loss_steps(self, data_loader: DataLoader):
        step_loss_list = []
        for step, (batch_x, batch_y) in enumerate(data_loader, start=1):
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

            output = self.model(batch_x)
            step_loss = self.loss_func(output, batch_y)
            step_loss_list.append(step_loss.item())
        mean_step_loss = torch.mean(torch.tensor(step_loss_list))
        return mean_step_loss.item()

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              metrics: list = None, epochs: int = 1,
              val_freq=1, callbacks: list = None):

        # todo print train params

        metrics = [] if not metrics else metrics
        for m in metrics:
            if m not in self.metric_func_lib.keys():
                raise ValueError("Arg 'metrics' is invalid: {}".format(list(self.metric_func_lib.keys())))

        callbacks = {str(cs): cs for cs in callbacks} if callbacks else []
        callbacks_func = {}
        for k, v in callbacks.items():
            callbacks_func[k] = v
            monitor = v.monitor
            if monitor not in metrics:
                metrics.append(monitor)

        metric_func = {}
        for m in metrics:
            if 'val' in m:
                metric_func[m] = (self.metric_func_lib[m], val_loader)
            else:
                metric_func[m] = (self.metric_func_lib[m], train_loader)

        train_log, logs = [], {item: [] for item in metrics}
        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))
            epoch_start = time.time()

            train_loss = self.train_loss_steps(train_loader)
            train_log.append(train_loss)

            for metric in metrics:
                if not ('val' in metric and e % val_freq != 0):
                    func = metric_func[metric][0]
                    arg = metric_func[metric][1]
                    metric_res = func(arg)
                    logs[metric].append(metric_res)

            print_train_log(logs, train_loss, epoch_start, e, val_freq)

            if callbacks:
                if 'best_saving' in callbacks:
                    callbacks['best_saving'].best_save(self.model, logs)
                elif 'early_stopping' in callbacks:
                    if callbacks['early_stopping'].early_stop(logs):
                        break

        logs['loss'] = train_log
        return logs
