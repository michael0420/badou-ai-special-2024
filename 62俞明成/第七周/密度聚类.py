from sklearn.cluster import DBSCAN
from sklearn import datasets
from matplotlib import pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
plt.rcParams['font.sans-serif'] = ['SimHei']

iris = datasets.load_iris()
X = iris.data[:, :4]


# 密度聚类算法
# eps:一个点要成为另一个点的邻域的最大距离
# min_samples:最小邻域点数
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
# 获取每个样本的簇标签,-1为噪声
label_pred = dbscan.labels_
# print(label_pred)

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == -1]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c="yellow", marker='x', label='噪声')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()
