"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: new_test.py
@time: 2020/11/25 14:00
@description: 
"""

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

from directpt import fit
import directpt.module as me
from directpt.functional import correct
from directpt.callback import BestSaving


# y_true = torch.tensor([[1, 0, 0], [1, 0, 0], [1, 0, 0],
#                        [0, 1, 0], [0, 1, 0], [0, 0, 1]])
#
# y_pre = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0],
#                       [0, 1, 0], [0, 1, 0], [0, 0, 1]])
#
# accuracy = me.MCAccuracy()
# acc = accuracy(y_pre, y_true)
# print()


l_relu = torch.nn.LeakyReLU()


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


# ------ 数据 ------
x_p = torch.ones(5000, 256).float()
x_n = torch.zeros(5000, 256).float()
x = torch.cat((x_p, x_n))

y_p = torch.ones(5000, 1).float()
y_n = torch.zeros(5000, 1).float()
y = torch.cat((y_p, y_n))

train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=20, shuffle=False)

# ------ 模型 ------
# 创建 模型
gen = Generator()
# 创建 优化器
opt = torch.optim.SGD(params=gen.parameters(), lr=1e-2, momentum=0.9)
# 创建 损失函数
metrics = torch.nn.MSELoss()
acc = correct

# ------ 框架训练 ------
# 创建训练器
trainer = fit.Trainer(gen, opt, acc, metrics, 'cpu')
# 回调函数
best_saving = BestSaving('save_path', monitor='val_loss', check_freq='epoch')
# 开始训练
trainer.train(train_loader, train_loader, epochs=10, val_freq=1,
              metrics=['loss', 'val_loss', 'acc', 'val_acc'], callbacks=[best_saving])

print()
