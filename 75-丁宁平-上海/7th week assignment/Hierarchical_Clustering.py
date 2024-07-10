# coding=utf-8

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

data = [[1, 2],
        [3, 2],
        [4, 4],
        [1, 2],
        [1, 3]]

# method = 'ward' (沃德方差最小化算法)
temp = linkage(data, method='ward')

#  当criterion为’distance’时，t值代表了绝对的差值，
#  如果小于这个差值，两个数据将会被合并，当大于这个差值，两个数据将会被分开。
rs = fcluster(temp, 4, criterion='distance')

fig = plt.figure(figsize=(6, 3))
ddg = dendrogram(temp)
print(temp)
plt.show()