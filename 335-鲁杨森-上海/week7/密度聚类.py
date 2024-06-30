import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from  sklearn.cluster import DBSCAN

#加载数据
iris = datasets.load_iris()
#print(iris.data)
x = iris.data[:,:4] #取前4列特征
#print(x.shape)

#原数据分布图 取前两列
# plt.scatter(x[:,0],x[:,1],c="red",marker='o',label='see')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend(loc=2)
# plt.show()

#eps 是两个样本被视为邻居的最大距离，min_samples 是一个点被视为核心点所需要的最小邻居数。
dbscan = DBSCAN(eps=0.5,min_samples=7)
#输入数据集
dbscan.fit(x)
label_pred = dbscan.labels_#label_pred 存储了每个样本的聚类标签。噪声点的标签为 -1。
#print(label_pred)
# 检查标签个数
unique_labels = np.unique(label_pred)
print("Unique labels:", unique_labels) #[-1,0,1]

x0 = x[label_pred == 0]
x1 = x[label_pred == 1]

#每次调用 plt.scatter 都会将新的散点添加到当前的图表中，而不会覆盖之前的散点。这样，你可以在一个图表中叠加多个数据集，每个数据集可以有不同的颜色、形状和标签。
#提取x的第一个和第二个特征分别作为x值，y值
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2) #设置图例的位置 loc=2表示左上角
plt.show()
