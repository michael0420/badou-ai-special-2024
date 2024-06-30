import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

if __name__ == '__main__':
    # 生成随机样本
    x, y = make_blobs(n_samples=1000, centers=4, random_state=40)
    # 使用凝聚层次聚类算法进行聚类
    clustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    clustering.fit(x, y)
    labels = clustering.labels_
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
