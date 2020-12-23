import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class LogPrinter:

    def __init__(self, epochs, total_step, val_freq):
        self.epochs = epochs
        self.total_step = total_step
        self.val_freq = val_freq

    def epoch_start_log(self, e: int):
        print('Epoch {}/{}'.format(e, self.epochs))

    def epoch_log(self, logs: dict, epoch_start: float, e_idx: int):
        """
        打印训练日志

        logs: 日志
        train_loss: 当前训练损失值
        epoch_start: 当前 epoch 开始时间
        e_idx: 当前进行的 epoch 次数
        val_freq: 计算验证的频率
        """
        msg = ''
        for m, v in logs.items():
            if not ('val' in m and e_idx % self.val_freq != 0):
                msg += ' - {}: {:.4f}'.format(m, v[-1])

        cost_time = time.time() - epoch_start
        if cost_time > 59:
            m, s = divmod(cost_time, 60)
            cost_time = f'{m}:{s}'
        elif cost_time > 3599:
            h, m = divmod(cost_time, 60)
            m, s = divmod(m, 60)
            cost_time = f'{h}:{m}:{s}'
        else:
            cost_time = f'{str(cost_time):.0}'

        print('{} {}'.format(cost_time, msg))

    def step_log(self, step: int, step_loss: float, step_acc: float):
        total_step = self.total_step
        past = int(step / total_step * 29)
        bar = '=' * past + '>' + '.' * (29 - past)
        pad_len = ' ' * (len(str(total_step)) - len(str(step))) + str(step)
        msg = '\r{}/{} [{}] - loss: {:.4f}'.format(pad_len, total_step, bar, step_loss)
        if step_acc is not None:
            msg += ' - acc: {:.4f}'.format(step_acc)
        print(msg, end='', flush=True)

    def epoch_end_log(self, epoch_start: float, epoch_loss: float, epoch_acc: float, val=False):
        total_step = self.total_step

        cost_time = time.time() - epoch_start
        if cost_time > 59:
            m, s = divmod(cost_time, 60)
            cost_time = f'{m}:{s}'
        elif cost_time > 3599:
            h, m = divmod(cost_time, 60)
            m, s = divmod(m, 60)
            cost_time = f'{h}:{m}:{s}'
        else:
            cost_time = f'{cost_time:.0f}s'

        msg = '\r{}/{} [{}] - ATA: {} - loss: {:.4f}'.format(total_step, total_step, '=' * 30, cost_time, epoch_loss)
        if epoch_acc is not None:
            msg += ' - acc: {:.4f}'.format(epoch_acc)
        if val:
            print(msg, end='', flush=True)
        else:
            print(msg)

    def add_val_log(self, logs):

        val_loss, val_acc = logs.get('val_loss'), logs.get('val_acc')
        msg = ''
        if val_loss:
            msg += ' - val_loss: {:.4f}'.format(val_loss[-1])
        if val_acc:
            msg += ' - val_acc: {:.4f}'.format(val_acc[-1])
        print(msg)


def model_summary(model, input_size, device, batch_size=-1):
    device = device.lower()

    print("-" * 64)
    if device == "gpu" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        print('{:>20}  {:>25}'.format('Training on', 'GPU'))
    else:
        dtype = torch.FloatTensor
        print('{:>20}  {:>25}'.format('Training on', 'CPU'))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # print(type(x[0]))

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-" * 64)
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("=" * 64)
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("=" * 64)
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("-" * 64)
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("-" * 64)
