import matplotlib.pyplot as plt
from  sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
# #表示我们只取特征空间中的4个维度
X= iris.data[:,:4]
print(X.shape)
dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_
#绘制结果
X0 = X[label_pred == 0]
X1 = X[label_pred == 1]
X2 = X[label_pred == 2]

plt.scatter(X0[:,0],X0[:,1],c='red',marker='o',label='Class 0')
plt.scatter(X1[:,0],X1[:,1],c='green',marker='*',label='Class 1')
plt.scatter(X2[:,0],X2[:,1],c='blue',marker='+',label='Class 2')
plt.rcParams['font.sans-serif']=['SimHei']
plt.xlabel("长")
plt.ylabel("宽")
# plt.legend(loc=1)
plt.show()