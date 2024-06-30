from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from matplotlib import pyplot as plt
'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
3.metric仅当输入是观测值数组（不是距离矩阵）时使用。指定用于计算观测值之间距离的度量标准。
'''
X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, method='single', metric='euclidean')  # 基于方差最小化的聚类方法

'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
3.criterion是聚类形成的准则
'distance'：使用指定的距离阈值t作为聚类间距离的最大值。任何距离小于或等于t的聚类都将被合并。
'maxclust'：使用指定的聚类数量t。函数将返回不超过t个聚类的划分。
'''
f = fcluster(Z, 2, 'maxclust')
fig = plt.figure(figsize=(5, 3))
dendrogram(Z)
print(Z)
print('--------------------------')
print(f)
plt.show()
