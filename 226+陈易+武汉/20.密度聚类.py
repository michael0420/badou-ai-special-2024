import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()   # 鸢尾花数据
X = iris.data[:, :4]            # 只取特征空间中的4个维度
print(X)

# 绘制数据分布图
"""
plt.scatter()用于绘制散点图,
x：横轴上的数据数组。
y：纵轴上的数据数组。
s：点的大小（默认值为 20）。
c：点的颜色，可以是字符串表示的颜色名称或表示颜色的数字。
marker：点的标记样式，例如 ‘o’ 表示圆点，‘s’ 表示方块。
alpha：点的透明度，取值范围为 0 到 1。
label：点的标签，用于图例中的标识
"""
plt.scatter(X[:,0], X[:,1], c="red", marker="o", label="see")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc=2)
plt.show()

# DBSCAN()函数。eps：领域半径，min_samples：领域半径内的最少点数
dbscan = DBSCAN(eps=0.4, min_samples=6)
dbscan.fit(X)
label_pred = dbscan.labels_

# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
plt.scatter(x0[:, 0], x0[:,1], c="red", marker="o",label="label0")
plt.scatter(x1[:, 0], x1[:,1], c="green", marker="*",label="label1")
plt.scatter(x2[:, 0], x2[:,1], c="blue", marker="+",label="label2")
plt.scatter(x3[:, 0], x3[:,1], c="yellow", marker="s",label="label3")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc=3)   # plt.legend() 将自动识别数据系列的标签，并在图表中添加图例
plt.show()