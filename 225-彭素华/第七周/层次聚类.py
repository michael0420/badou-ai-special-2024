from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt

'''

linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。



除了'single'，还有其他常用的链接方法，包括：

Complete Linkage (method='complete')：也称为最大链接方法，它定义两个簇之间的距离为两个簇中最远两个点之间的距离。

Average Linkage (method='average')：使用两个簇中所有点之间的平均距离来定义簇之间的距离。

Centroid Linkage (method='centroid')：使用两个簇的质心之间的距离来定义簇之间的距离。

Ward Linkage (method='ward')：使用Ward's 方法来定义簇之间的距离，该方法最小化合并两个簇时的总方差增加量。

'''

'''

fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。

'''

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, 'ward')
f = fcluster(Z, 10, 'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()
