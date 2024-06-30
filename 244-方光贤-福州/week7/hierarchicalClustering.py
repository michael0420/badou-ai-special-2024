from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt

# 定义数据集
X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]

# 计算欧氏距离矩阵
dist_matrix = squareform(pdist(X, 'euclidean'))

# 计算链接矩阵 使用ward方式 通过方差大小来选择先合并的两个聚类
Z = linkage(dist_matrix, 'ward')

# 从链接矩阵中提取扁平化的聚类（例如 最多3个聚类）
f = fcluster(Z, 3, criterion='maxclust')

# 绘制树状图
fig = plt.figure(figsize=(5, 3))
dendrogram(Z)

# 显示链接矩阵和树状图
print(Z)
plt.show()