from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
"""
linkage(y, method='single', metric='euclidean', optimal_ordering=False):
    1.y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
        若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。  
    2. method是指计算类间距离的方法。
        single 最小距离
        complete 最大距离
        average 平均距离
        weighted 加权距离
        centroid 中心距离
        median 和中心距离相似
        ward 基于Ward方差最小化算法

fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
    1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
    2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”

"""


X = [[1],[2],[1],[99],[98],[99],[5],[65],[64]]

# X = [[1,2], [3,2], [4,4], [1,2], [1,3]]
Z = linkage(X, 'ward')
print(Z)
f = fcluster(Z, 4, 'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()



