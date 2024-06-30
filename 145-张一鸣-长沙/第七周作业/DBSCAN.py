# coding = utf-8

'''
        实现密度聚类
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

# 加载4个维度的鸢尾花数据
iris = datasets.load_iris()
X = iris.data
print(X.shape)
print(X)

# 原始数据绘图展示
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='iris')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

# 声明密度聚类函数
db = DBSCAN(eps=0.4, min_samples=9)
# 执行
db.fit_predict(X)

# 通过labels_属性获取每个样本点的簇标签，-1通常表示噪声点
labels = db.labels_
print(type(labels))
print(labels)

# 绘制结果
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
x0 = X[labels == 0]
x1 = X[labels == 1]
x2 = X[labels == 2]
x3 = X[labels == -1]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='聚类1')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='聚类2')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='聚类3')
# plt.scatter(x3[:, 0], x3[:, 1], c="black", marker='x', label='噪音')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

