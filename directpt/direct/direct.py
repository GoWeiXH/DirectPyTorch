from torch.utils import data

from ..fit.train import Trainer
from ..exception import NoCompileError
from ..data import train_test_split, to_data_loader


class Direct:

    def __init__(self, device='cpu'):
        self.device = device
        self.network = None
        self.trainer = None

    def compile(self, network, loss_func, optimizer, pre_threshold: float = 0.5):
        self.network = network
        self.trainer = Trainer(self.network, optimizer, loss_func,
                               device=self.device, pre_threshold=pre_threshold)

    def fit(self, x_data=None, y_label=None,
            metrics: list = None,
            epochs: int = 1, batch_size=1, val_freq=1,
            test_size: float = 0.2,
            test_data: tuple = None, test_batch_size: int = None,
            callbacks: list = None,
            random_seed=None, shuffle=False, num_workers=1):

        if not self.trainer:
            raise NoCompileError()

        # todo print train params

        if x_data is not None and y_label is not None:
            if test_data is None:
                train_loader, test_loader = train_test_split(
                    x_data, y_label,
                    test_size, batch_size,
                    random_seed, shuffle,
                    num_workers)
            else:
                train_set = data.TensorDataset(x_data, y_label)
                test_set = data.TensorDataset(test_data[0], test_data[1])
                train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)
                test_loader = data.DataLoader(test_set, batch_size=test_batch_size,
                                              shuffle=shuffle, num_workers=num_workers)
        else:
            raise TypeError('NoneType: x_data and y_label is None')

        self.trainer.train(train_loader, test_loader, metrics, epochs, val_freq, callbacks)
