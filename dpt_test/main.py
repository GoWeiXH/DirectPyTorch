import os
import sys

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

cur_path = os.path.abspath(__file__)
for d in range(3):
    cur_path = os.path.dirname(cur_path)
sys.path.append(cur_path)

import directpt.module as me
from directpt.direct import Direct
from directpt.callback import BestSaving
from directpt.data import train_test_split


# y_pre = torch.tensor([1, 1, 1, 0, 0, 0])
# y_true = torch.tensor([1, 1, 1, 1, 1, 0])

# y_pre = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
# y_true = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
#
# method = me.Recall(multi_class=True)
# res = method(y_pre, y_true)
# print()

# accuracy = me.MCAccuracy()
# acc = accuracy(y_pre, y_true)
# print()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(256, 128),
            # me.Linear(128, 10, activation_func=l_relu, activation_param={'negative_slope': 0.2})
            me.Linear(128, 1, activation='sigmoid', activation_param={'inplace': True})
        )

    def forward(self, x):
        x = self.gen(x)
        return x


if __name__ == '__main__':
    # ------ 数据 ------
    x_p = torch.ones(5000, 256).float()
    x_n = torch.zeros(5000, 256).float()
    x_data = torch.cat((x_p, x_n))

    y_p = torch.ones(5000, 1).float()
    y_n = torch.zeros(5000, 1).float()
    y_label = torch.cat((y_p, y_n))

    # ------ 模型 ------
    # 创建 模型
    gen = Generator()
    # 创建 优化器
    opt = torch.optim.SGD(params=gen.parameters(), lr=1e-2, momentum=0.9)
    # 创建 损失函数
    loss = torch.nn.MSELoss()
    # 回调函数
    best_saving = BestSaving('best_model/save_path.pt', monitor='val_loss', check_freq='epoch')

    direct = Direct()
    direct.compile(gen, loss, opt, pre_threshold=0.5)
    direct.fit(x_data, y_label, metrics=['val_acc'],
               epochs=20, batch_size=200,
               callbacks=[best_saving])
