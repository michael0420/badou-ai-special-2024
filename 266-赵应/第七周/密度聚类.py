from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 生成样本数据
    x, y = make_blobs(1000, centers=4, random_state=50)
    clustering = DBSCAN(eps=0.5)
    clustering.fit(x, y)
    labels = clustering.labels_
    # 分类
    # 获取分类结果并画图
    cluster1 = x[labels == 0]
    cluster2 = x[labels == 1]
    cluster3 = x[labels == 2]
    cluster4 = x[labels == 3]
    plt.scatter(cluster1[:, 0], cluster1[:, 1], marker="+", label='cluster1')
    plt.scatter(cluster2[:, 0], cluster2[:, 1], marker="*", label='cluster2')
    plt.scatter(cluster3[:, 0], cluster3[:, 1], marker="o", label='cluster3')
    plt.scatter(cluster4[:, 0], cluster4[:, 1], marker="s", label='cluster4')
    plt.show()
