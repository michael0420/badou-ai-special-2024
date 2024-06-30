"""
实现密度聚类
Author： zsj
"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

# 获取样本数据
iris = datasets.load_iris()
X = iris.data[:, :4]
# 设置eps=0.4（即邻域半径），min_samples=9（成为核心对象所需的邻域中的样本数）
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
# 打好标签的集合
label_pred = dbscan.labels_
print(label_pred)
# 根据不同的簇绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='2')
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.legend()
plt.show()
