from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

# 定义数据集
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]

# 使用 ward 方法进行层次聚类，得到聚类树的连接矩阵
Z = linkage(X, 'ward')

# 将数据集进行层次聚类，并指定阈值为 4，返回每个样本的聚类标签
f = fcluster(Z, 4, 'distance')

# 创建一个大小为 (5, 3) 的图像对象
fig = plt.figure(figsize=(5, 3))

# 绘制聚类树状图
dn = dendrogram(Z)
print(Z)
plt.show()
