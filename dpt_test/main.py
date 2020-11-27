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
from directpt.callback import BestSaving
import directpt.module as me

# y_pre = torch.randn(100, 10, requires_grad=True)
# y_pre[:, 1:3] = 1
#
# y_true = torch.zeros(100, 10)
# y_true[:, 0:3] = 1
#
# loss = y_pre * y_true
# print()


l_relu = torch.nn.LeakyReLU()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(256, 128),
            me.Linear(128, 10, activation_func=l_relu, activation_param={'negative_slope': 0.2})
            # me.Linear(128, 10, activation='sigmoid', activation_param={'inplace': True})
        )

    def forward(self, x):
        x = self.gen(x)
        return x


# ------ 数据 ------
x = torch.randn(10, 256).float()
y = torch.randn(10, 10).float()
train_set = TensorDataset(x, y)
train_loader = DataLoader(train_set, batch_size=2, shuffle=False)

# ------ 模型 ------
# 创建 模型
gen = Generator()
# 创建 优化器
opt = torch.optim.SGD(params=gen.parameters(), lr=1e-5, momentum=0.9)
# 创建 损失函数
metrics = torch.nn.MSELoss()

# ------ 框架训练 ------
# 创建训练器
trainer = fit.Trainer(gen, opt, metrics, 'cpu')
# 回调函数
best_saving = BestSaving('save_path', monitor='val_loss', check_freq='epoch')
# 开始训练
trainer.train(train_loader, train_loader, epochs=10, val_freq=1,
              metrics=['loss', 'val_loss'], callbacks=[best_saving])

print()
