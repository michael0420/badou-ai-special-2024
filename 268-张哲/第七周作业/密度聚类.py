import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

#获取数据集
iris = datasets.load_iris()
# 只取特征空间中的4个维度
X = iris.data[:,:4]
#初始化密度聚类函数
dbscan = DBSCAN(eps=0.4,min_samples=9) #eps:半径，min_samples:半径内最少数据
dbscan.fit(X)
label_pre = dbscan.labels_

#绘制结果
x0 = X[label_pre == 0]
x1 = X[label_pre == 1]
x2 = X[label_pre == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()



