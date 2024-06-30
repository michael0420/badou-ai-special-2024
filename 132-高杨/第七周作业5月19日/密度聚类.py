from sklearn import datasets
import matplotlib.pyplot as plt
import sklearn.cluster as clu
import cv2

iris = datasets.load_iris()
print(iris.data.shape)
X = iris.data[:,:]

dbscan = clu.DBSCAN(eps=0.4,min_samples=9)
dbscan.fit(X)
labels = dbscan.labels_


x0 = X[labels==0]
x1 = X[labels==1]
x2 = X[labels==2]

plt.scatter(x0[:,0],x0[:,1],c='red',marker='x')
plt.scatter(x1[:,0],x1[:,1],c='green',marker='*')
plt.scatter(x2[:,0],x2[:,1],c='blue',marker='+')

plt.xlabel('speal length')
plt.xlabel('speal height')
plt.show()



