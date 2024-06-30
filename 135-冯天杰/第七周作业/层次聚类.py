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

from sklearn import datasets
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
import matplotlib.pyplot as plt
import numpy as np
from numpy import float_

X = datasets.load_iris()

X = X.data[:,:2]* 100
print(np.shape(X))

Y = linkage(X, 'ward')

print(type(Y))
f = fcluster(Y, 3, 'distance')

f = np.array(f,dtype=np.double)
print(f)
print(np.shape(f))
fig = plt.figure(figsize=(50,10))
plt.xticks(np.arange(min(Y[0]),max(Y[0])+1,5))
d = dendrogram(Y)
plt.show()