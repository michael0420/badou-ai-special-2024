import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt


def DS(X, eps, min_samples):
    # 初始化DBSCAN对象
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    # 拟合数据
    dbscan.fit(X)
    # 获取聚类结果
    label_pred = dbscan.labels_

    # 绘制聚类结果
    plot_clusters(X, label_pred)


def hierarchical_clustering(X, method='ward', criterion='distance', t=None):
    # 层次聚类
    Z = linkage(X, method)
    if t is None:
        t = max(Z[:, 2]) / 2  # 使用中间值作为默认阈值
    # 根据阈值获取聚类结果
    labels = fcluster(Z, t, criterion=criterion)
    plt.figure(figsize=(5, 3))
    # 绘制树状图
    dn = dendrogram(Z)
    plt.show()
    return labels


def plot_clusters(X, labels):
    # 获取唯一的标签值
    unique_labels = np.unique(labels)
    # 生成颜色映射
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    # 绘制聚类结果
    for label, color in zip(unique_labels, colors):
        cluster = X[labels == label]
        plt.scatter(cluster[:, 0], cluster[:, 1], color=color, marker='o', label='label {}'.format(label))

    plt.xlabel('length')
    plt.ylabel('width')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X = iris.data[:, :4]  # 表示只取特征空间中的前四个维度
    print(X.shape)
    # # 打印数据的前五行
    # print(X[:5])
    # # 打印数据的前两列
    # print(X[:, :2])

    # 执行DBSCAN聚类
    DS(X, 0.5, 9)
    # # Example of usage
    # X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
    labels = hierarchical_clustering(X, method='ward', criterion='distance', t=4)
    print(labels)