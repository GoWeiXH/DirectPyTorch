import torch
import numpy as np
from torch.utils import data


def transform_to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).float()
    elif isinstance(x, list):
        return torch.tensor(x)
    else:
        raise TypeError
        # raise TypeError("Data type is invalid, should be np.ndarray, torch.Tensor or list")


def train_test_split(x_data, y_label, test_size, batch_size,
                     seed=None, shuffle: bool = True, num_workers=1,
                     source_data=None):
    # 数据格式转换
    x_data = transform_to_tensor(x_data)
    y_label = transform_to_tensor(y_label)

    # 检测 x_data 与 y_label 数据长度是否相等
    x_len, y_len = x_data.shape[0], y_label.shape[0]
    assert x_len == y_len, "X_data len and Y_label len is unequal (x: {}, y: {})".format(x_len, y_len)

    data_idx = list(range(x_len))
    if seed is not None:
        np.random.seed(seed)
        shuffle = True
    if shuffle:
        np.random.shuffle(data_idx)

    # 划分数据
    x_y_data_set = data.TensorDataset(x_data, y_label)
    divide = int(x_len * (1 - test_size))

    train_set = data.Subset(x_y_data_set, data_idx[:divide])
    test_set = data.Subset(x_y_data_set, data_idx[divide:])
    train_loader = to_data_loader(data_set=train_set, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers)
    test_loader = to_data_loader(data_set=test_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=num_workers)

    # 获取对应的源数据
    if source_data:
        source = np.array(source_data)[data_idx]
        train_source = source[:divide]
        test_source = source[divide:]
        return train_loader, test_loader, train_source, test_source

    return train_loader, test_loader


def to_data_loader(x=None, y=None, data_set=None, batch_size=1024, shuffle=False, num_workers=1):
    if x is not None and y is not None:
        data_set = data.TensorDataset(x, y)
    if not data_set:
        raise TypeError('x, y or data_set should not be None.')
    return data.DataLoader(dataset=data_set, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers)
