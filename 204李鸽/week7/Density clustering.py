import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from  sklearn.cluster import DBSCAN

iris = datasets.load_iris()        # 从 sklearn 库中的 datasets 模块加载了鸢尾花数据集，将数据存储在 iris 变量中
X = iris.data[:, :4]               # 第一个 : 表示取所有行，第二个 :4 前四列，前4维度特征
print(X.shape)
# 绘制数据分布图
'''
plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see') 
# X[:, 0]：表示取变量 X 中所有行的第一列数据作为 x 轴数据。
# X[:, 1]：表示取变量 X 中所有行的第二列数据作为 y 轴数据。
# c="red"：指定散点的颜色为红色。你也可以使用其他颜色，比如 "blue"、"green"、"yellow" 等，或者使用十六进制值如 "#FF0000"。
# marker='o'：指定散点的形状为圆形。你可以使用其他形状，如 "s" 表示方形，"^" 表示三角形等。
# label='see'：为这组数据点添加一个标签，用于图例中显示。 
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)     #loc=2表示设置图例的位置在左上，loc=1表示右上角，loc=4表示右下角，loc=3表示左下角
plt.show()  
'''
dbscan = DBSCAN(eps=0.4, min_samples=9)
# DBSCAN：Density-Based Spatial Clustering of Applications with Noise即聚类
# eps=0.4表示定义邻域半径为0.4，min_samples=9表示定义一个核心点所需要的样本数目为9
dbscan.fit(X)                   # 将数据集X传入DBSCAN对象中
label_pred = dbscan.labels_     # 获取每个样本点的聚类标签，在DBSCAN算法中，每个数据点将被分配一个标签，指示它属于哪个簇。

'''具体来说，在DBSCAN算法中：
核心点（core point）将被分配一个正整数标签，表示它所属的簇的标记。
边界点（border point）将被分配与其邻近的核心点相同的标签。
噪声点（noise point）将被分配标签-1，表示它们被视为异常值或不属于任何簇。'''

# 绘制结果
x0 = X[label_pred == 0]   # 从原始数据集X中筛选出标签为0的数据点
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')     #绘制标签为0的数据点，横坐标为x0的第一列数据，纵坐标为x0的第二列数据，颜色为红色，形状为圆形，图例为'label0'
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')                                               #设置x轴标签为'sepal length'。
plt.ylabel('sepal width')
plt.legend(loc=2)                                                        #添加图例，位置在左上角
plt.show()