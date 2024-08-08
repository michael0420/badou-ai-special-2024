import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

"""
matplotlib.pyplot 用于绘图。
numpy 是Python的一个库，用于进行科学计算。
datasets 从 sklearn 中导入，用于加载各种数据集。
DBSCAN 是从 sklearn.cluster 中导入的，它是一个基于密度的聚类算法。
"""
iris = datasets.load_iris()
X = iris.data[:, :4]
print(X.shape)

"""
使用 datasets.load_iris() 加载鸢尾花数据集。
X 存储了数据集的特征部分，这里我们只取了前4个维度（即花萼长度、花萼宽度、花瓣长度、花瓣宽度）。
"""

dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

"""
创建一个 DBSCAN 实例，设置 eps=0.4（邻域大小）和 min_samples=9（成为核心点所需的最小样本数）。
使用 .fit(X) 对数据集 X 进行聚类。
label_pred 存储了每个样本的聚类标签。

"""

# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

"""
这三行代码通过布尔索引（boolean indexing）从X中选择了分别对应预测标签0、1和2的数据点。
这意味着x0、x1和x2现在分别包含了所有被预测为类别0、1和2的数据点。
接下来，使用plt.scatter函数分别为这三个类别的数据点绘制散点图：
"""
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')

"""
在每次调用plt.scatter时，都指定了数据点的x坐标（x0[:, 0]、x1[:, 0]、x2[:, 0]）、y坐标（x0[:, 1]、x1[:, 1]、x2[:, 1]）、
颜色（'red'、'green'、'blue'）、形状（'o'、'*'、'+'）和标签（'label0'、'label1'、'label2'）。
这样，就可以在同一个图表中看到三个不同类别的数据点，并且它们各自的颜色和形状都有所不同，便于区分。
"""
plt.xlabel('sepal length')
plt.ylabel('sepal width')
"""
这两行代码分别设置了x轴和y轴的标签为“sepal length”（萼片长度）和“sepal width”（萼片宽度），这有助于理解图表中的数据。
"""
plt.legend(loc="upper right")
plt.show()

