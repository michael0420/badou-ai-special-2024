from sklearn.cluster import DBSCAN  # 从sklearn库导入DBSCAN聚类算法
from sklearn import datasets        # 从sklearn库导入数据集模块
import matplotlib.pyplot as plt


iris = datasets.load_iris()     # 加载Iris数据集
x = iris.data[:,:4]     # 取特征空间中的前4个维度数据

dbscan = DBSCAN(eps=0.4, min_samples=9)     # 初始化DBSCAN算法，eps=0.4表示邻域半径，min_samples=9表示核心点的最小样本数
dbscan.fit(x)       # 对数据集x进行聚类
label_pred = dbscan.labels_     # 获取聚类标签，每个样本对应一个标签，表示其所属的聚类


# 将聚类标签的样本提取出来
x0 = x[label_pred==0]
x1 = x[label_pred==1]
x2 = x[label_pred==2]

plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')

# 设置横轴标签为花萼长度（sepal length）
plt.xlabel('sepal length')
# 设置纵轴标签为花萼宽度（sepal width）
plt.ylabel('sepal width')
# 设置图例位置为左上角
plt.legend(loc=2)
plt.show()

