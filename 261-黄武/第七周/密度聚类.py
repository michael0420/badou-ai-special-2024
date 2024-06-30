
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#生成模拟数据

data,labels = make_blobs(n_samples=200,centers=4,random_state=20)

#设置DBSCAN参数

eps = 0.5  #邻域半径
min_samples = 5 #最小样本数

#应用DBSCAN算法

dbscan = DBSCAN(eps = eps,min_samples=min_samples)
dbscan.fit(data)

#获取聚类结果

cluster_lables = dbscan.labels_

#打印聚类结果

print('聚类标签：',cluster_lables)

#绘制散点图

plt.scatter(data[:,0],data[:,1],s=10,c=cluster_lables,cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBSCAN Clustering')
plt.show()
