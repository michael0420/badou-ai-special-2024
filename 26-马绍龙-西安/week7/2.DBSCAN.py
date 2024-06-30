import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

'''
load_iris()返回一个字典-like对象，其中包含了以下属性：
    data：一个二维数组，包含了150个样本的四个特征值。
    target：一个一维数组，表示150个样本对应的鸢尾花种类的标签。
    target_names：一个包含三个类别名称的列表。
    DESCR：一个字符串，提供了数据集的详细描述。
'''
iris = datasets.load_iris()
X = iris.data[:, :4]  # 切片取特征空间中的前4列，即4个维度
print(X.shape)

# 绘制原始数据分布图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='原始数据')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()

dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_  # -1：表示数据点被认为是“噪声”或“边缘点”，它们没有被分配到任何簇中。
print('聚类结果：')
print(label_pred)

# 绘制密度聚类结果
xx = X[label_pred == -1]
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(xx[:, 0], xx[:, 1], c="black", marker='v', label='噪声')
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')

plt.xlabel('sepal length')  # 添加坐标轴标签
plt.ylabel('sepal width')
plt.legend(loc=2)  # 将图例放置在上右位置

plt.show()
