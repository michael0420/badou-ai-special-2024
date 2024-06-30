import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons, make_blobs

# 生成模拟数据
np.random.seed(1)
X1, _ = make_moons(n_samples=300, noise=0.05)
X2, _ = make_blobs(n_samples=100, centers=[[3, 3]], cluster_std=0.5)
X = np.vstack((X1, X2))

# 使用DBSCAN进行聚类
db = DBSCAN(eps=0.2, min_samples=5).fit(X)
labels = db.labels_

# 获取核心样本和噪声样本
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# 绘制结果
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 噪声点
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14, label=f'Cluster {k}')

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6)

plt.title(f'Estimated number of clusters: {n_clusters_}')
plt.legend()
plt.show()
