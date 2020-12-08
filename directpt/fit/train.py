"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: train.py
@time: 2020/11/25 16:02
@description: 
"""

import time
from types import FunctionType

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader


class Trainer:
    """
    训练器
    """

    def __init__(self, model: Module, optimizer: Optimizer,
                 acc: nn.Module, loss: nn.Module, device=None):
        self.model = model
        self.acc_func = acc
        self.loss_func = loss
        self.optimizer = optimizer

        self.main_device, self.device, self.is_parallel = self.set_train_device(device)

    def set_train_device(self, device) -> (str, str, bool):
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

    def train_steps(self, data_loader: DataLoader):
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
        mean_step_loss = torch.mean(torch.tensor(step_loss_list))
        return mean_step_loss.item()

    def test_steps(self, data_loader: DataLoader):
        step_loss_list = []
        for step, (batch_x, batch_y) in enumerate(data_loader, start=1):
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

            output = self.model(batch_x)
            step_loss = self.loss_func(output, batch_y)
            step_loss_list.append(step_loss.item())
        mean_step_loss = torch.mean(torch.tensor(step_loss_list))
        return mean_step_loss.item()

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              metrics=('loss',), epochs: int = 1,
              val_freq=1, callbacks=None):

        logs = dict().fromkeys(metrics)
        for item in logs.keys():
            logs[item] = []

        metric_func = {
            'loss': (self.train_steps, (train_loader,)),
            'val_loss': (self.test_steps, (val_loader,)),
            'acc': (self.acc_steps, (train_loader,)),
            'val_acc': (self.acc_steps, (val_loader,))
        }
        for m in metrics:
            if m not in metric_func.keys():
                raise ValueError("Arg 'metrics' is invalid: {}".format(metric_func.keys()))

        if callbacks:
            callbacks = {str(cs): cs for cs in callbacks}

        for e in range(1, epochs + 1):
            print('Epoch {}/{}'.format(e, epochs))
            epoch_start = time.time()

            # todo val_freq

            for metric in metrics:
                func = metric_func[metric][0]
                args = metric_func[metric][1]
                metric_res = func(*args)
                logs[metric].append(metric_res)

            loss_msg = ' - '.join(('{}: {:.6f}'.format(m, v[-1]) for m, v in logs.items() if 'loss' in m))
            acc_msg = ' - '.join(('{}: {:.2f}'.format(m, v[-1]) for m, v in logs.items() if 'acc' in m))
            print(' - {:.0f}s - {} - {}'.format(time.time() - epoch_start, loss_msg, acc_msg))

            if callbacks:
                if 'best_saving' in callbacks:
                    callbacks['best_saving'].best_save(self.model, logs)
                elif 'early_stopping' in callbacks:
                    if callbacks['early_stopping'].early_stop(logs):
                        break
