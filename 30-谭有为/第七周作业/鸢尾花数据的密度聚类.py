import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

datas=datasets.load_iris()  #加载鸢尾花数据集
print(datas)
X=datas.data   #取鸢尾花数据集中的data
print(X,X.shape)

#绘制鸢尾花数据的散点图
plt.scatter(X[:,0],X[:,1],c='red',marker='*')
plt.show()

dbscan=DBSCAN(eps=0.4,min_samples=10)  #设置密度聚类的邻域半径和最小点数目
dbscan.fit(X) #密度聚类
rult=dbscan.labels_   #获取密度聚类返回的标签
print(rult)

#画图
x0 = X[rult == 0]
x1 = X[rult == 1]
x2 = X[rult == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Iris Density clustering')
plt.legend(loc='best')
plt.show()

