import os
import sys

import torch
import torch.nn as nn

cur_path = os.path.abspath(__file__)
for d in range(3):
    cur_path = os.path.dirname(cur_path)
sys.path.append(cur_path)

from directpt.direct import Direct
from directpt.callbacks.callbacks import BestSaving
from directpt.metrics import recall_precision_fscore, binary_accuracy, multi_class_accuracy


# y_pre = torch.tensor([1, 1, 1, 0, 0, 0])
# y_true = torch.tensor([1, 1, 1, 1, 1, 0])

# y_pre = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
# y_true = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

# y_pre = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1]])
# y_true = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])


# r, p, f = recall_precision_fscore(y_pre, y_true, multi=False)
# acc = binary_accuracy(y_pre, y_true)
# print()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.gen(x)
        return x


if __name__ == '__main__':
    # ------ 数据 ------
    x_p = torch.ones(500, 256).float()
    x_n = torch.zeros(500, 256).float()
    x_data = torch.cat((x_p, x_n))

    y_1 = torch.tensor([0, 1]).repeat(500, 1).float()
    y_2 = torch.tensor([1, 0]).repeat(500, 1).float()
    y_label = torch.cat((y_1, y_2))
    # y_p = torch.ones(500, 1).float()
    # y_n = torch.zeros(500, 1).float()
    # y_label = torch.cat((y_p, y_n))

    # ------ 模型 ------
    # 创建 模型
    gen = Generator()
    # 创建 优化器
    opt = torch.optim.SGD(params=gen.parameters(), lr=1e-4, momentum=0.9)
    # 创建 损失函数
    # loss = torch.nn.MSELoss()
    loss = torch.nn.BCELoss()
    # 回调函数
    best_saving = BestSaving('best_model/save_path.pt', monitor='val_loss', check_freq='epoch')

    direct = Direct()
    direct.compile(gen, loss, opt, threshold=0.5)
    direct.fit(x_data, y_label, metrics=['acc'],
               epochs=20, batch_size=10,
               callbacks=[best_saving])
