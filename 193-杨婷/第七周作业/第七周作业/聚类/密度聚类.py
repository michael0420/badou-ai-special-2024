import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
# print(iris)
X = iris.data[:, :4]  # 只取特征空间4维度
print(X.shape)

# 绘制数据分布图
'''
plt.scatter(X[:, 0], X[:, 1], c='red', marker='o', label='data')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)  # loc=2通常意味着图例会放在图的左上角
plt.show()
'''
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)  # fit方法会计算每个样本的类别标签，但不会直接返回这些标签，而是存储在DBSCAN对象的内部。
label_pred = dbscan.labels_  # 从DBSCAN对象中获取聚类标签
print(label_pred)  # -1标签表示那些被认为是噪声点的样本，它们没有被分配到任何簇中。
# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()