# encoding utf-8
# zym 2022.1.14
import time


def postfix_generater():
    return str(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()))
