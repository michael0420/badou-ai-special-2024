"""
层次聚类 不用预设层次关系，甚至可以发现层次关系，但是计算复复杂度高，可能聚类成链状
"""
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X =iris.data[:, :4] #取四个维度

# fig = plt.figure(figsize=(10,5))
# plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend(loc='best')
# plt.show()

dbscan = DBSCAN(eps=0.4,min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]


plt.figure(figsize=(10,10))
plt.scatter(x0[:,0], x0[:,1], c="red", marker='o', label='01')
plt.scatter(x1[:,0], x1[:,1], c="green", marker='*', label='02')
plt.scatter(x2[:,0], x2[:,1], c="blue", marker='x', label='03')


plt.title("DBSCAN")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc='best')
plt.show()