import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4]  # 所有行（使用冒号:）和前4列（使用:4）  只取特征空间中的4个维度
# iris.data[::4] # 每4行切一次片

print(X.shape)
'''
#原始图像
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()
'''
dbscan =DBSCAN(eps=0.4,min_samples=9)   #半径，最少点数
dbscan.fit(X)
label_pred = dbscan.labels_

#绘图
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

plt.rcParams['font.sans-serif']=['SimHei']

plt.scatter(x0[:,0],x0[:,1],c='red',marker='*',label='第一类')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='+', label='第二类')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='o', label='第三类')

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=1)   # 0：自动找， 1：右上角  2：左上角
plt.show()
