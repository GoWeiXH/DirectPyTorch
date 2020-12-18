import time


class Printer:

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
        if step_acc:
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
        if epoch_acc:
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
