# encoding utf-8
# zym 2021.1.24
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from visual.ImgUtils import postfix_generater

matplotlib.use('Agg')


def pca_plot(data, label, name=None):
    '''
        目前仅能处理2各类别，主要是画图部分问题
        data np数据格式 -> n*dim
        label list格式 -> n*bool 1 or 0
    '''
    #标准化
    scaler = StandardScaler()
    scaler.fit(data)
    scaled = scaler.transform(data)

    #PCA
    pca = PCA().fit(scaled)

    pc = pca.transform(scaled)
    pc1 = pc[:,0]
    pc2 = pc[:,1]

    #画出主成分
    plt.figure(figsize=(10,10))

    colour = ['#ff2121' if y == 1 else '#2176ff' for y in label]
    plt.scatter(pc1,pc2 ,c=colour,edgecolors='#000000')
    plt.ylabel("Glucose",size=20)
    plt.xlabel('Age',size=20)
    plt.yticks(size=12)
    plt.xticks(size=12)
    plt.xlabel('PC1')
    img_name = "pca" + postfix_generater()
    if name is not None:
        img_name = name + img_name
    img_name = "img/" + img_name
    plt.savefig(img_name)


if __name__ == "__main__":
    A = np.random.rand(10, 128)
    A_label = [0 for _ in range(5)] + [1 for _ in range(5)]
    pca_plot(A, A_label)

