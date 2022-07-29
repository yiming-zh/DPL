# coidng: utf-8
# ZYM 2020/11/16


import imp
import seaborn as sns
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from visual.ImgUtils import postfix_generater
matplotlib.use('Agg')


def visualization(
        attn_weight, x_label, y_label, groud_truth=None, name=None):
    '''
    Plot the attention model heatmap
    attn_weight: [len_x, len_y]   Tensor格式
    x_label: [x_len]
    y_label: [y_len]
    '''
    attn_weight = attn_weight.detach().numpy()
    fig, ax = plt.subplots(figsize=(15, 10))
    max_value = max(attn_weight.max(), 0.7)
    min_value = min(attn_weight.min(), 0.7)
    attn_weight = (attn_weight-min_value) / (max_value - min_value)
    ax = sns.heatmap(attn_weight, ax=ax, vmax=1, vmin=0)
    # 处理标签
    xticks = range(0, len(x_label))
    ax.set_xticks(xticks, minor=False)
    ax.set_xticklabels(x_label, minor=False, rotation=0)
    yticks = range(0, len(y_label))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(y_label, minor=False, rotation=0)
    ax.invert_yaxis()
    ax.grid(True)
    # 增加ground truth便于对比
    if groud_truth is not None:
        plt.title(groud_truth)
    # 保存图片
    if name is None:
        name = postfix_generater()
        name = 'img/' + name
    fig.savefig(name)
    plt.close(fig)
