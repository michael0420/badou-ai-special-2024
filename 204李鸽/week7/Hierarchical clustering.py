from scipy.cluster.hierarchy import dendrogram, linkage,fcluster      # 这三个是层次聚类模块中的函数
from matplotlib import pyplot as plt
'''
linkage(y, method='single', metric='euclidean', optimal_ordering=False)
y：包含观测数据的二维数组，表示要进行聚类的数据集。
method：指定用于计算聚类链接的方法。常见的方法包括：
'single'：最小距离法
'complete'：最大距离法
'average'：平均距离法
'ward'：Ward 方法
metric：指定用于计算观测数据之间距离的度量标准。可以是很多不同的度量标准，例如：
'euclidean'：欧氏距离
'cityblock'：曼哈顿距离
'cosine'：余弦距离
optimal_ordering：是否启用优化排序。如果设置为 True，将尝试使用更高效的算法来确定观测数据点的顺序。

fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None, count=2, monocd=None)
Z：聚类链接矩阵，即由 linkage 函数生成的矩阵，包含了每次合并簇的信息。
t：阈值，用于控制聚类的数量。一般是根据聚类链接矩阵的距离信息来确定。
criterion：指定聚类的标准。常见的标准包括：
'inconsistent'：根据相对高度不一致性确定聚类数量。
'distance'：基于给定阈值确定聚类数量。
depth：用于计算聚类数量的深度。
R：树的多重比较水平。
monocrit：单值聚类标准。
count：每个节点的观测数据数量。
monocd：单值聚类深度。'''

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, 'ward')
f = fcluster(Z, 4, 'distance')
fig = plt.figure(figsize=(5, 3))       # 指定图形的尺寸为宽 5 英寸，高 3 英寸
dn = dendrogram(Z)                     # 聚类链接矩阵。生成树状图
print(Z)                               # 打印聚类链接矩阵 Z 的内容，查看聚类过程中的距离信息
plt.show()                             # 当不传入参数时，plt.show() 默认会显示当前所有已经创建的图形,这是因为 Matplotlib 会跟踪