import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

# 加载iris数据集
iris = datasets.load_iris()
# 取特征空间中的前4个维度
X = iris.data[:, :4]

# 初始化DBSCAN对象,
# eps确定点与点之间的半径, 会搜索任何点半径为0.4区域的所有邻近点
# min_samples eps邻近内必须要有9个点才会被确定为核心点
dbscan = DBSCAN(eps=0.4, min_samples=9)
# 对数据进行拟合，执行DBSCAN聚类
dbscan.fit(X)
# 获取聚类标签
label_pred = dbscan.labels_

# 根据聚类标签，筛选出各个聚类对应的数据点
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

# 使用matplotlib绘制散点图，不同簇使用不同的颜色和标记
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')

# 设置x轴和y轴的标签
plt.xlabel('sepal length')
plt.ylabel('sepal width')

# 设置图例的位置
plt.legend(loc=2)

# 显示图像
plt.show()
