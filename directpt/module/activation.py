import torch.nn as nn

from ..exception import ActivationError


class Activation(nn.Module):
    """
    激活函数类

    Args:
        func_name: 激活函数名称，当前支持：relu、leaky_relu、sigmoid、softmax
        param: 激活函数中包含的可调整参数

    Examples:
        >>> act = Activation('lrelu', 0.2)

    """

    def __init__(self, func_name, param=None, **options):
        super(Activation, self).__init__()
        self.func_name = func_name
        self.param = param
        self.activation = self.get_func(**options)

    def get_func(self, **options):
        if self.func_name == 'relu':
            func = nn.ReLU(**options)
        elif self.func_name == 'lrelu':
            func = nn.LeakyReLU(self.param) if self.param else nn.LeakyReLU()
        elif self.func_name == 'sigmoid':
            func = nn.Sigmoid()
        elif self.func_name == 'softmax':
            func = nn.Softmax(self.param) if self.param else nn.Softmax()
        else:
            raise ActivationError(self.func_name)
        return func

    def forward(self, input_tensor):
        return self.activation(input_tensor)

    def __bool__(self):
        return bool(self.activation)
