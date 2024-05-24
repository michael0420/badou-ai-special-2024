import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

# 加载奕尾花数据
bunch = datasets.load_iris()
# 表示知趣奕尾花数据的四个维度
bunch_data_ = bunch.data[:, :4]
print(bunch_data_.shape)

dbscan = DBSCAN(eps=0.4, min_samples=9)
dbscan.fit(bunch_data_)
labels_data = dbscan.labels_

# 绘制结果
x0 = bunch_data_[labels_data == 0]
x1 = bunch_data_[labels_data == 1]
x2 = bunch_data_[labels_data == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend(loc=2)
plt.show()
