"""
@author: 207-xujinlan
密度聚类算法
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from  sklearn.cluster import DBSCAN

# 1.加载鸢尾花数据
data_iris = datasets.load_iris().data[:, :4]
# 2.模型声明
dbscan = DBSCAN(eps=0.4, min_samples=9)
# 3.模型训练
dbscan.fit(data_iris)

# 4.聚类结果展示
label_pred = dbscan.labels_
iris0 = data_iris[label_pred == 0]
iris1 = data_iris[label_pred == 1]
iris2 = data_iris[label_pred == 2]
plt.scatter(iris0[:, 0], iris0[:, 1], c="blue", marker='o', label='iris0')
plt.scatter(iris1[:, 0], iris1[:, 1], c="red", marker='*', label='iris1')
plt.scatter(iris2[:, 0], iris2[:, 1], c="yellow", marker='+', label='iris2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
