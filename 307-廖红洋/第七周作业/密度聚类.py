# -*- coding: utf-8 -*-
"""
@File    :   密度聚类.py
@Time    :   2024/06/02 15:10:06
@Author  :   廖红洋 
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

# plt.ion()
# 获取数据，为sklearn上的体能训练数据
data = datasets.load_linnerud()
X = data.data[:, :3]

# 绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker="o", label="see")
plt.xlabel("Chains")
plt.ylabel("Situps")
plt.legend(loc=2)
plt.show()

# 进行密度聚类
dbscan = DBSCAN(
    eps=30, min_samples=2
)  # 调整聚类参数，半径eps内点多于3就标记为一个类中心点，若其他点范围内包含此点且满足中心点条件，则也成为此类中心点
dbscan.fit(X)  # 密度聚类方法
label_pred = dbscan.labels_  # 存储聚类类别

# 绘制结果,这里其实可以优化，因为不知道具体会聚出多少类，因此x0这种类别信息可以根据label聚类的结果动态分配数量
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker="o", label="label0")
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker="*", label="label1")
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker="+", label="label2")
plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker="o", label="label3")
plt.xlabel("Chains")
plt.ylabel("Situps")
plt.legend(loc=2)
plt.show()
