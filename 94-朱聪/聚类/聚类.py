###cluster.py
# 导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。 ward是方差最小化

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

# X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3], [7, 5], [1, 4], [5, 3], [2, 6]]
# Z = linkage(X, 'ward')
# f = fcluster(Z, 6, 'distance')
# fig = plt.figure(figsize=(5, 3))
# dn = dendrogram(Z) # 将Z中的聚类树绘制为树状图
# print(Z)
# plt.show()


###  密度聚类

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
print(X.shape)
# 绘制数据分布图
'''
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')  
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)  
plt.show()  
'''

dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_ # 标签为-1代表是噪声，具体有几类是根据参数改变的，只能先看结果才知道有几类

# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
# x3 = X[label_pred == 3]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
# plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='x', label='label3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()