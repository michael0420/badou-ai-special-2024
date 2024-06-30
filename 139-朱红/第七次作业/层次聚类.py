from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

# 计算类间距离，采用ward方差最小化算法
Z = linkage(X, 'ward')
fig = plt.figure(figsize=(15, 10))
# 将层次聚类编码为树状图的链接矩阵
dn = dendrogram(Z)
plt.show()

Z = linkage(X, 'single')
fig = plt.figure(figsize=(15, 10))
dn = dendrogram(Z)
plt.show()