import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

# 加载鸢尾花数据集
iris = datasets.load_iris()
# 取出鸢尾花数据集所有数据行的前4列数据
x = iris.data[:, :4]

# 基于密度的聚类算法，eps:以某个样本为中心，在半径为eps的范围内所能找到的样本数目
# min_samples：核心对象所需要的邻域内样本数目的最小值
dbscan = DBSCAN(eps=0.4, min_samples=9)
# 根据数据x进行聚类分析
dbscan.fit(x)
# 样本标签
label_pred = dbscan.labels_

# 绘制结果图像
x0 = x[label_pred == 0]
x1 = x[label_pred == 1]
x2 = x[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()
