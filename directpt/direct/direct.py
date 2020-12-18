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

    def fit(self, x_data, y_label,
            train_loader: data.DataLoader = None,
            test_loader: data.DataLoader = None,
            metrics: list = None,
            epochs: int = 1, batch_size=1, val_freq=1,
            test_size: float = 0.2,
            test_x=None, test_y=None,
            callbacks: list = None,
            random_seed=None, shuffle=False, num_workers=1):

        if not self.trainer:
            raise NoCompileError()

        # todo print train params

        if test_x is None and test_y is None and test_size is not None:
            if x_data is not None and y_label is not None:
                train_loader, test_loader = train_test_split(
                    x_data, y_label, test_size, batch_size, random_seed, shuffle, num_workers)
        else:
            # todo
            ...

        self.trainer.train(train_loader, test_loader, metrics, epochs, val_freq, callbacks)
