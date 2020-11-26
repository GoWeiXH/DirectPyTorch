"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: train.py
@time: 2020/11/25 16:02
@description: 
"""

from types import FunctionType

import torch
import torch.nn as nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader


class Trainer:

    def __init__(self, model: Module, optimizer: Optimizer,
                 loss: FunctionType, device=None):
        self.model = model
        self.loss_func = loss
        self.optimizer = optimizer

        self.main_device, self.device, self.is_parallel = self.set_train_device(device)

    def set_train_device(self, device) -> (str, str, bool):
        if device == 'gpu':
            if torch.cuda.is_available():
                main_device = 'cuda:0'
            else:
                # todo gpu is unavailable
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

    def calculation(self, data_loader: DataLoader, is_train: bool):
        step_loss_list = []
        for step, (batch_x, batch_y) in enumerate(data_loader, start=1):
            batch_x, batch_y = batch_x.to(self.main_device), batch_y.to(self.main_device)

            output = self.model(batch_x)
            step_loss = self.loss_func(output, batch_y)
            step_loss_list.append(step_loss.item())

            if is_train:
                self.optimizer.zero_grad()
                step_loss.backward()
                self.optimizer.step()

            # todo log loss, train True, test False

        mean_step_loss = torch.mean(torch.tensor(step_loss_list))
        return mean_step_loss.item()

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              metrics=('loss',), epochs: int = 1,
              val_freq=1, callbacks=None):

        logs = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

        metric_func = {
            'loss': (self.calculation, (train_loader, True)),
            'val_loss': (self.calculation, (val_loader, False)),
            'acc': (lambda x: x, (train_loader,)),
            'val_acc': (lambda x: x, (val_loader,))
        }
        for m in metrics:
            if m not in metric_func.keys():
                raise ValueError("Arg 'metrics' is invalid: {}".format(metrics))

        if callbacks:
            callbacks = {str(cs): cs for cs in callbacks}

        train_total_step = len(train_loader)

        for e in range(1, epochs + 1):

            for metric in metrics:
                func = metric_func[metric][0]
                args = metric_func[metric][1]
                metric_res = func(*args)
                logs[metric].append(metric_res)

            if callbacks:
                if 'best_saving' in callbacks:
                    callbacks['best_saving'].save_best(self.model, logs)
                elif 'early_stopping' in callbacks:
                    if callbacks['early_stopping'].early_stop(logs):
                        break
