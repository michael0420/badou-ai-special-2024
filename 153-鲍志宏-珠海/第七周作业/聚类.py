#####层次聚类#####

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt


#导入数据
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]

# 计算层次聚类的连接矩阵
Z = linkage(X, 'ward')

# 生成聚类标签
f = fcluster(Z,4,'distance')

# 绘制树状图
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)


print(Z)
plt.show()

#####密度聚类#####

import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import datasets 
from  sklearn.cluster import DBSCAN

# 加载数据集，取特征空间中的4个维度
iris = datasets.load_iris() 
X = iris.data[:, :4]  
print(X.shape)


# 创建DBSCAN模型
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X) 
label_pred = dbscan.labels_
 
# 绘制结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')  
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')  
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')  
plt.xlabel('sepal length')  
plt.ylabel('sepal width')  
plt.legend(loc=2)  
plt.show()  
