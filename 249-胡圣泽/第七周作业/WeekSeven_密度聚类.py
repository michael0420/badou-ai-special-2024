import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

"""
对密度聚类原理进行理解：探索未访问点的ε邻域，邻域内点足够多，则形成一个类；点不够，则标记该点为噪声
notion：被标记为噪声的点可能处于某点的ε邻域内且该邻域内点足够多，形成为一个类。
"""

iris = datasets.load_iris()
X = iris.data[:, :4]
print(X.shape)

dbscan = DBSCAN(eps=0.4, min_samples=9)  #类似于实例化，声明聚类算法实例对象；DBS(邻域ε, 最少点数minPts)
dbscan.fit(X) #实例对象.fit(data)把数据吃进去
label_pred = dbscan.labels_

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()