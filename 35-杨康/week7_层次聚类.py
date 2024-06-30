from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。single:最短距离 comlete:最大距离 average:平均距离 centroid:中心距离 ward：离差平方和距离

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。 
3.criterion可以设置为'distance'，表示使用距离作为阈值来决定聚类的数量。
例如，fcluster(Z, t=3, criterion='distance')表示从给定的链接矩阵Z中形成的层次聚类中，使用距离作为标准来形成平面聚类，其中t=3意味着当两个聚类之间的距离小于3时，它们将被合并。
'''
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, method='centroid')
f = fcluster(Z,4,'distance')
plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
plt.show()
