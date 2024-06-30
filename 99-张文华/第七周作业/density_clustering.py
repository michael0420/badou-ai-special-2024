'''
实现密度聚类
'''

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data[:, 0:4]
print(data)

dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(data)
label_pred = dbscan.labels_

print(label_pred)

# 绘制图像
x0 = data[label_pred == 0]
x1 = data[label_pred == 1]
x2 = data[label_pred == 2]
x3 = data[label_pred == -1]      # -1表示噪声点
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c="black", marker='*', label='label3')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()