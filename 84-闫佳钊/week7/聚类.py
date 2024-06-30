import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import DBSCAN
from sklearn import datasets

# ===层次聚类
# X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
# Z = linkage(X, method='ward')
# fcluster(Z, 4, 'distance')
# dendrogram(Z)
# # plt.figure()
# plt.show()
# ===密度聚类
iris = datasets.load_iris()
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
print(X.shape)
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=1)
plt.show()
