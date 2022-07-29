# encoding utf-8
# zym 2021.1.24
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from visual.ImgUtils import postfix_generater
matplotlib.use('Agg')


def tsne_plot(data, index_list=None, name_list=None, img_path=None):
    '''
    data 为 list 格式
    index_list 为 每个组结尾处index的list, 从0开始
    name_list 为 legend 的labels
    '''
    data = np.array(data)
    TSNE = manifold.TSNE(n_components=2, init='pca')
    embed_data = TSNE.fit_transform(data)

    # normalization
    data_max, data_min = embed_data.max(0), embed_data.min(0)
    embed_data = (embed_data-data_min) / (data_max-data_min)

    # plot
    fig = plt.figure(figsize=(8, 8))
    plt_list = []
    if index_list is None:
        plt.scatter(embed_data[:, 0], embed_data[:, 1])
    else:
        for i in range(len(index_list)-1):
            begin, end = index_list[i], index_list[i+1]
            plt_list.append(
                plt.scatter(
                    embed_data[begin: end, 0], embed_data[begin: end, 1]
                )
            )
        plt.legend(handles=plt_list, labels=name_list, loc='best')
    plt.xticks([])
    plt.yticks([])
    # 保存图片
    if img_path is None:
        img_path = postfix_generater()
        img_path = 'img/' + img_path
    fig.savefig(img_path)
    plt.close(fig)


if __name__ == '__main__':
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    import torch
    X = torch.tensor(X)
    tsne_plot(
        X, index_list=[0, 2, 4], name_list=['haha', '????'], img_path='test'
    )
