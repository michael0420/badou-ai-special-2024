import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
X = iris.data[:, :4]  # 表示我们只取特征空间中的4个维度

dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(X)
label_pred = dbscan.labels_

num_clusters = len(np.unique(label_pred[label_pred != -1]))  # 排除标签为-1的噪声点
print("Total number of clusters:", num_clusters)

categories = ["red", "green", "blue", "yellow", "purple", "pink"]

for i in range(num_clusters):
    xi = X[label_pred == i]
    plt.scatter(xi[:, 0], xi[:, 1], c=categories[i % len(categories)], marker='o', label='label' + str(i))

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()
