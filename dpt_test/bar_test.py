"""
@version: V1.0
@author: weizhenhao
@mail: weizhenhao@bjgoodwill.com
@file: bar_test.py
@time: 2020/12/7 15:31
@description: 
"""

import time

total_step = 20

for step in range(1, total_step+1):
    past = int(step / total_step * 29)
    bar = '=' * past + '>' + '.' * (29 - past)

    pad_len = ' ' * (len(str(total_step)) - len(str(step))) + str(step)
    print('\r{}/{} [{}]'.format(pad_len, total_step, bar), end='', flush=True)
    time.sleep(0.5)
bar = '=' * 30
print('\r{}/{} [{}]'.format(total_step, total_step, bar), end='', flush=True)
