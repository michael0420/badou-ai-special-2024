# coding=utf-8

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
data = iris.data[:, :4]  # 只取特征空间中的4个维度
# print(data)
print(data.shape)

# 绘制数据分布图
# plt.scatter(data[:, 0], data[:, 1], c="red", marker='o', label='see')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend(loc=1)
# plt.show()

dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(data)
label_pred = dbscan.labels_


# 绘制结果
x0 = data[label_pred == 0]
x1 = data[label_pred == 1]
x2 = data[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c='red', marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c='green', marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c='blue', marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=1)
plt.show()
