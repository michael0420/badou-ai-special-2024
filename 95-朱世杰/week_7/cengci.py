"""
实现层次类聚
@Author：zsj
"""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

# 数据
X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]

# 层次类聚，'ward'是连接方法的一种，称为最小方差法
# 返回结果是层次聚类过程的链表，记录了每次聚合的距离和簇的信息
Z = linkage(X, 'ward')
# 根据距离对聚类进行标记
f = fcluster(Z, 4, 'distance')
print(f)
# 创建窗口
fig = plt.figure(figsize=(5, 3))
# 绘制图形
dn = dendrogram(Z)
print(Z)
plt.show()
