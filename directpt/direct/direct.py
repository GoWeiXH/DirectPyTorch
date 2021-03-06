from typing import Union

import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils import data

from ..fit import Trainer
from ..utils import model_summary
from .._exception import NoCompileError
from ..data import train_test_split, transform_to_tensor


class Direct:

    def __init__(self, device: Union[str, list] = 'cpu'):
        self.device = device
        self.main_device = ''
        self.is_parallel = False
        self.network = None
        self.trainer = None
        self.loss_func = None
        self.optimizer = None
        self.multi = False

    def compile(self, network: Module, loss_func, optimizer, threshold: float = 0.5):
        self.network = network
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.main_device, self.device, self.is_parallel = self.set_train_device(self.device)
        self.trainer = Trainer(self.network, optimizer, loss_func,
                               main_device=self.main_device, threshold=threshold)

    def set_train_device(self, device: str) -> (str, str, bool):
        main_device = 'cpu'
        if torch.cuda.is_available():
            if isinstance(device, str):
                if device == 'gpu':
                    main_device = 'cuda:0'
                elif device.startswith('cuda'):
                    main_device = device
            elif isinstance(device, list):
                main_device = 'cuda:{}'.format(device[0])

        is_parallel = False
        if isinstance(device, list):
            if len(device) > 1 and torch.cuda.device_count() > 1:
                self.network = nn.DataParallel(self.network, device_ids=device)
                is_parallel = True

        self.network.to(main_device)

        device = 'cpu' if main_device == 'cpu' else 'gpu'
        return main_device, device, is_parallel

    def set_multi_flag(self, y_label):
        try:
            y_label = transform_to_tensor(y_label)
            if len(y_label.shape) == 1:
                y_label = y_label.reshape(-1, 1)
            if y_label.shape[1] > 1:
                self.multi = True
        except TypeError:
            output_dim = list(self.network.parameters())[-1].shape[0]
            if output_dim > 1:
                self.multi = True

    def fit(self, x_data=None, y_label=None,
            train_loader: data.DataLoader = None,
            test_loader: data.DataLoader = None,
            metrics: list = None,
            epochs: int = 1, batch_size=1, val_freq=1,
            test_size: float = 0.2,
            test_data: tuple = None, test_batch_size: int = None,
            callbacks: list = None,
            random_seed=None, shuffle=True, num_workers=1,
            verbose: bool = False):

        if not self.trainer:
            raise NoCompileError()

        # todo 网络、训练参数

        if x_data is not None and y_label is not None:
            # 如果未指定测试数据，则以 test_size 划分测试数据
            if test_data is None:
                train_loader, test_loader = train_test_split(
                    x_data, y_label,
                    test_size, batch_size,
                    random_seed, shuffle,
                    num_workers
                )

            # 如果指定测试数据，则无需划分
            else:
                train_set = data.TensorDataset(x_data, y_label)
                test_set = data.TensorDataset(test_data[0], test_data[1])
                train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)
                test_loader = data.DataLoader(test_set, batch_size=test_batch_size,
                                              shuffle=shuffle, num_workers=num_workers)

        elif train_loader is not None and test_loader is not None:
            train_loader = train_loader
            test_loader = test_loader

        else:
            raise TypeError('NoneType: x_data and y_label is None')

        if verbose:
            model_summary(self.network, (x_data.shape[1],), self.device)

        self.set_multi_flag(y_label)
        self.trainer.train(train_loader, test_loader,
                           metrics, epochs, self.multi,
                           val_freq, callbacks)

    @staticmethod
    def load_model(model: nn.Module, model_file_path: str, device: str) -> (nn.Module, dict):

        model.to(device)

        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint.pop('model_state_dict'))
        model.eval()

        return model, checkpoint
