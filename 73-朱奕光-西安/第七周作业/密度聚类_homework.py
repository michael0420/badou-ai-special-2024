import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
print(iris.feature_names)
data = iris.data
db = DBSCAN(eps=0.4, min_samples=9)
labels = db.fit_predict(data)
print(labels)
c1 = data[labels == 0]
c2 = data[labels == 1]
c3 = data[labels == 2]
plt.scatter([n[0] for n in c1], [n[1] for n in c1], c='red', marker='d')
plt.scatter([n[0] for n in c2], [n[1] for n in c2], c='blue', marker='2')
plt.scatter([n[0] for n in c3], [n[1] for n in c3], c='green', marker='p')
plt.show()
