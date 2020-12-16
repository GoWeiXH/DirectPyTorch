import time


def print_train_log(logs: dict, train_loss: float, epoch_start: float, e_idx: int, val_freq: int):
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
        if not ('val' in m and e_idx % val_freq != 0):
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

    print('{} - loss: {:.4f} {}'.format(
        cost_time, train_loss, msg))
