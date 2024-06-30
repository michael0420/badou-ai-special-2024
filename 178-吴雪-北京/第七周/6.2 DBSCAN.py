"""
Density clustering (DBSCAN)
"""
from sklearn import datasets
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

iris = datasets.load_iris()  # 导入鸢尾花数据集
X = iris.data[:,: 4]  # 表示只取特征空间中的4个维度（获取所有行的前4列）
print(X.shape)
print(X)

# eps-临近半径r （ε-领域）
# min_samples-最小样本数 （形成高密度区域所需要的最少点数：也就是在ε-领域内的点数）
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)  # # 对DBSCAN模块进行训练
# dbscan = DBSCAN(eps=0.4, min_samples=9).fit(X) # 导入DBSCAN模块进行训练
label_pred = dbscan.labels_  # labels为每个数据的簇标签

# 绘制数据分布图
x0 = X[label_pred == 0]  # 获取聚类标签等于0的话，则赋值给x0
x1 = X[label_pred == 1]  # 获取聚类标签等于1的话，则赋值给x1
x2 = X[label_pred == 2]  # 获取聚类标签等于2的话，则赋值给x2
plt.scatter(x0[:, 0], x0[:, 1], c='r', marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c='g', marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c='b', marker='+', label='label2')
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc=2)  # loc=2 这个位置就是4象项中的第二象项，也就是左上角
plt.show()
