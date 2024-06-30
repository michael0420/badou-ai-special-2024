import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4] #只取四个维度
print(X.shape)

dbScan = DBSCAN(eps=0.4, min_samples=9)
dbScan.fit(X)
labels = dbScan.labels_
print(labels)

#绘制结果
X0 = X[labels == 0]
X1 = X[labels == 1]
X2 = X[labels == 2]
plt.scatter(X0[:, 0], X0[:, 1], c="red", marker='o', label='label0')
plt.scatter(X1[:, 0], X1[:, 1], c="green", marker='*', label='label1')
plt.scatter(X2[:, 0], X2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('length')
plt.ylabel('width')
plt.legend(loc=2)
plt.show()