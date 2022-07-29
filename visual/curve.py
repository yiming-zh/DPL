# encoding utf-8
# zym 2021.1.24
import time
import matplotlib
import matplotlib.pyplot as plt
from visual.ImgUtils import postfix_generater
matplotlib.use('Agg')


def plot_train_curve(
    x_label,
    y_label,
    epoch_num,
    train_vals,
    valid_vals=None,
    img_path=None
):
    # set figsize
    if valid_vals is not None:
        name = ['aspect', 'sentence']
    else:
        name = ['valid']
    fig = plt.figure(figsize=(6, 4))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    x_vals = [i+1 for i in range(epoch_num)]
    train, = plt.plot(x_vals, train_vals, linestyle='-', color='blue')
    if valid_vals is None:
        plt.legend(handles=[train], labels=name, loc='best')
    else:
        valid, = plt.plot(x_vals, valid_vals, linestyle='-', color='red')
        plt.legend(handles=[train, valid], labels=name, loc='best')
    # 保存图片
    if img_path is None:
        img_path = postfix_generater()
        img_path = 'img/' + img_path
    fig.savefig(img_path)
    plt.close(fig)


if __name__ == "__main__":
    train_ans = [
        81.65, 86.17, 89.36, 90.24, 93.77, 96.35, 98.26, 98.78, 99.12, 99.35
    ]
    valid_ans = [
        81.65, 82.17, 83.58, 84.33, 85.21, 84.27, 84.79, 84.10, 85.69, 86.90
    ]
    plot_train_curve(
        'epoch', 'acc', len(train_ans), train_ans, valid_ans, img_path='test'
    )
