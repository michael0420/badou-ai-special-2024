from sklearn import datasets
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

X = datasets.load_iris()
X = X.data[:, :4]
# print(X)

dbscan = DBSCAN(eps=0.4, min_samples= 9)
dbscan.fit(X)
labels = dbscan.labels_

a = X[labels == 0]
b = X[labels == 1]
c = X[labels == 2]
print(a)
plt.scatter(a[:,0],a[:,1], c='b', marker='x')
plt.scatter(b[:,0],b[:,1], c='g', marker='x')
plt.scatter(c[:,0],c[:,1], c='r', marker='x')

plt.show()