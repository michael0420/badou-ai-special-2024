'''
密度聚类DBSCAN，在一定半径r邻域内形成高密度区点数(含自己)，也能用于异常点检测
密度直达、密度可达
density-based spatial clustering of applications with noise 可在噪声空间数据库中聚类任意形状的高密度区
核心点：e邻域有规定点数；边界点：在其它核心点的邻域内，但不是核心点；噪声点：非核心点，且不在任意核心点e邻域内
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN

iris = datasets.load_iris()
# print(iris, iris.data, iris.target)  # iris含有的鸢尾花4个维度特征空间，以及对应标记
X = iris.data
# X = iris.data[:, :4]
# 老师的方法表示只取特征空间中的4个维度， [a:b, c:d]表示a~b行，c~d列
# 萼片sepal length (cm), sepal width (cm)；花瓣petal length (cm),petal width (cm)
# plt.subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], s=30, c='red', marker='*', label='sepal')
# plt.xlabel('sepal length'), plt.xticks([3, 4.5, 6, 7.5])
# plt.ylabel('petal length')
# plt.legend(loc='upper right')  # 显示图例，要在plt.scatter中注明label,也可以loc=2，直接规定位置
# plt.subplot(1, 2, 2)
# plt.scatter(X[:, 2], X[:, 3], s=30, c='green', marker='^')
# plt.show()

dbscan = DBSCAN(eps=0.4, min_samples=9)  # 声明参数,再放入数据
db_pred = dbscan.fit_predict(X)
# dbscan.fit(X)
# db_pred = dbscan.labels_  # 老师的方法
print(db_pred)
X0, X1, X2, X3 = X[db_pred == 0], X[db_pred == 1], X[db_pred == 2], X[db_pred == -1]
'''
-1表示噪声，绘图只展示了2个维度，可能看起来重合不是噪声，但在4个维度空间上就不一定
'''
print('X0是:\n{}\nX1是:\n{}\nX2是:\n{}\n噪声点标记X3是:\n{}\n'.format(X0, X1, X2, X3))
plt.scatter(X0[:, 0], X0[:, 1], s=30, c='red', label='X0', marker='p')
plt.scatter(X1[:, 0], X1[:, 1], s=30, c='green', label='X1', marker='>')
plt.scatter(X2[:, 0], X2[:, 1], s=30, c='blue', label='X2', marker='*')
plt.scatter(X3[:, 0], X3[:, 1], s=30, c='black', label='X3', marker='+')
plt.xlabel('sepal length'), plt.ylabel('sepal width'), plt.title('sepal length & width DBSCAN')
plt.legend(loc='upper right')
plt.legend(loc=2)
plt.show()

# 先利用PCA降维再绘制二维或三维图
ax3 = plt.axes(projection='3d')
ax3.scatter(X0[:, 0], X0[:, 1], X0[:, 2], color='red', marker='p')
ax3.scatter(X1[:, 0], X1[:, 1], X1[:, 2], color='green', marker='>')
ax3.scatter(X2[:, 0], X2[:, 1], X2[:, 2], color='blue', marker='*')
ax3.scatter(X3[:, 0], X3[:, 1], X3[:, 2], color='black', marker='+')
plt.show()
