'''

2实现密度聚类
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from  sklearn.cluster import DBSCAN

# 读取鸢尾花的数据，python数据库中自带的
iris=datasets.load_iris()
# 储存其中的一部分鸢尾花的数据
#表示我们只取特征空间中的4个维度
iris_data=iris.data[:,:4]
print(iris_data.shape)

# 使用DBSCAN密度聚类函数对鸢尾花进行聚类
# 先加载DBSCAN函数
iris_dbscan=DBSCAN(eps=0.4,min_samples=9)
# 对鸢尾花数据执行密度聚类
iris_dbscan.fit(iris_data)
# 获取聚类后的数据
iris_pred=iris_dbscan.labels_

# print(iris_pred)

# 对数据结果进行绘图显示
iris_pred0=iris_data[iris_pred==0]
iris_pred1=iris_data[iris_pred==1]
iris_pred2=iris_data[iris_pred==2]
plt.scatter(iris_pred0[:, 0], iris_pred0[:, 1], c="red", marker='o', label='label0')
plt.scatter(iris_pred1[:, 0], iris_pred1[:, 1], c="green", marker='*', label='label1')
plt.scatter(iris_pred2[:, 0], iris_pred2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()