from sklearn.cluster import DBSCAN
from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data[:,:4]
print(X.shape)
dbscan = DBSCAN(eps=0.4,min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_
print(label_pred)
x0 = X[label_pred==0]
x1 = X[label_pred==1]
x2 = X[label_pred==2]
#x3 = X[label_pred==-1]
plt.scatter(x0[:,0],x0[:,1],c='green',marker='x',label='label1')
plt.scatter(x1[:,0],x1[:,1],c='red',marker='.',label='label2')
plt.scatter(x2[:,0],x2[:,1],c='blue',marker='o',label='label3')
#plt.scatter(x3[:,0],x3[:,1],c='yellow',marker='D',label='label4')
plt.legend(loc=2)
plt.show()