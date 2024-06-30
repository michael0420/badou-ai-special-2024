# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
ourData = pd.read_csv('Mall_Customers.csv')
ourData.head()

#使用该数据集在Annual Income (k$)和Spending Score (1-100)列上实现我们的层次聚类模型
newData = ourData.iloc[:, [3, 4]].values
newData

import scipy.cluster.hierarchy as sch # 导入层次聚类算法
dendrogram = sch.dendrogram(sch.linkage(newData , method = 'ward')) # 使用树状图找到最佳聚类数
plt.title('Dendrogram') # 标题
plt.xlabel('Customers') # 横标签
plt.ylabel('Euclidean distances') # 纵标签
plt.show()

from sklearn.cluster import AgglomerativeClustering
# n_clusters为集群数，affinity指定用于计算距离的度量，linkage参数中的ward为离差平方和法
Agg_hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = Agg_hc.fit_predict(newData) # 训练数据

plt.scatter(newData[y_hc == 0, 0], newData[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1') # cluster 1
plt.scatter(newData[y_hc == 1, 0], newData[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2') # cluster 2
plt.scatter(newData[y_hc == 2, 0], newData[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3') # cluster 3
plt.scatter(newData[y_hc == 3, 0], newData[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')  #  cluster 4
plt.scatter(newData[y_hc == 4, 0], newData[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #  cluster 5

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
