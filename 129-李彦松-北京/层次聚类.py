###cluster.py
#导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import numpy as np

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def agglomerative_clustering(X, n_clusters):
    # 初始化每个点为一个簇
    clusters = [[point] for point in X]

    while len(clusters) > n_clusters:
        # 计算所有簇之间的距离
        distances = [[euclidean_distance(np.mean(cluster1, axis=0), np.mean(cluster2, axis=0))
                      for cluster2 in clusters] for cluster1 in clusters]

        # 找到距离最近的两个簇
        min_distance = float('inf')
        cluster1, cluster2 = 0, 0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if distances[i][j] < min_distance:
                    min_distance = distances[i][j]
                    cluster1, cluster2 = i, j

        # 合并两个簇
        clusters[cluster1].extend(clusters[cluster2])
        del clusters[cluster2]

    # 为每个点分配一个簇标签
    labels = np.zeros(len(X), dtype=int)
    for cluster_id, cluster in enumerate(clusters):
        for point in cluster:
            index = np.where((X == point).all(axis=1))[0][0]
            labels[index] = cluster_id

    return labels

# 使用函数进行层次聚类
X = np.array([[1,2],[3,2],[4,4],[1,2],[1,3]])
labels = agglomerative_clustering(X, n_clusters=2)

# 将簇标签转换为颜色代码
colors = ['red' if label == 1 else 'blue' for label in labels]

plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.show()
