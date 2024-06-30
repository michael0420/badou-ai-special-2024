import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets

# # 加载鸢尾花的数据
# X=datasets.load_iris()
# # 取每个样本的前四个特征
# X=X.data[:,:4]
#
# d=DBSCAN(eps=0.4,min_samples=9)
# label_pred=d.fit_predict(X)
#
# x0=X[label_pred==0]
# x1=X[label_pred==1]
# x2=X[label_pred==2]
# plt.scatter(x0[:,0],x0[:,1],c='red',marker='*',label='label0')
# plt.scatter(x1[:,0],x1[:,1],c='black',marker='o',label='label1')
# plt.scatter(x2[:,0],x2[:,1],c='blue',marker='x',label='label2')
# plt.xlabel('length')
# plt.ylabel('width')
# plt.legend()
# plt.show()



X=datasets.load_iris()
X=X.data[:,:4]
d=DBSCAN(eps=0.4,min_samples=9)
label_pred=d.fit_predict(X)

x0=X[label_pred==0]
x1=X[label_pred==1]
x2=X[label_pred==2]

plt.scatter(x0[:,0],x0[:,1],c='red',marker='*',label='label0')
plt.scatter(x1[:,0],x1[:,1],c='black',marker='o',label='label1')
plt.scatter(x2[:,0],x2[:,1],c='blue',marker='x',label='label2')
plt.xlabel('length')
plt.ylabel('width')
plt.legend()
plt.show()