from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()  # 这样取鸢尾花的数据
X = iris.data[:, :4]  # #表示我们只取特征空间中的4个维度
print(X.shape)

# 声明
dbscan = DBSCAN(eps=0.4, min_samples=9)  # min_samples一个点周围有9个点才能聚成一类
# eps参数有什么作用？我刚刚试了0.2，0.3,0.4,0.5，效果确实不一样了，但是没看出什么规律来，什么时候该给多大的值
# 执行
dbscan.fit(X)
# 读取标签
label_pred = dbscan.labels_
print(label_pred)

# 画图
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
