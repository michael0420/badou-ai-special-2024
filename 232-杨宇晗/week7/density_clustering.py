import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数学库
from sklearn import datasets  # 从sklearn库导入数据集工具
from sklearn.cluster import DBSCAN  # 导入DBSCAN聚类算法

iris = datasets.load_iris()  # 加载iris数据集
X = iris.data[:, :4]  # 取数据集中的前四个特征用于聚类
print(X.shape)  # 打印数据的维度，以确认数据的结构

# 下面的代码块用来在需要时绘制数据点的分布图
'''
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')  # 绘制数据点
plt.xlabel('sepal length')  # 设置x轴标签
plt.ylabel('sepal width')  # 设置y轴标签
plt.legend(loc=2)  # 显示图例，位置在左上角
plt.show()  # 显示图形
'''

dbscan = DBSCAN(eps=0.4, min_samples=9)  # 创建DBSCAN聚类器，设置邻域大小和最小样本点数
dbscan.fit(X)  # 对数据进行聚类
label_pred = dbscan.labels_  # 获取聚类标签

# 根据聚类标签绘制聚类结果
x0 = X[label_pred == 0]  # 获取标签为0的点
x1 = X[label_pred == 1]  # 获取标签为1的点
x2 = X[label_pred == 2]  # 获取标签为2的点
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')  # 绘制标签为0的数据点
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  # 绘制标签为1的数据点
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')  # 绘制标签为2的数据点
plt.xlabel('sepal length')  # 设置x轴标签
plt.ylabel('sepal width')  # 设置y轴标签
plt.legend(loc=2)  # 显示图例，位置在左上角
plt.show()  # 显示图形
