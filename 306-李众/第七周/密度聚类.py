import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4]

dbscan = DBSCAN(eps=0.4, min_samples=9).fit(X)
label_pred = dbscan.labels_

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], marker="x",c="r")
plt.scatter(x1[:, 0], x1[:, 1], marker="o",c="g")
plt.scatter(x2[:, 0], x2[:, 1], marker="^",c="b")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
