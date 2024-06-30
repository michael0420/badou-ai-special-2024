from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

# 生成测试数据
X, y = make_circles(100, factor=0.5, noise=0.1)

# 创建DBSCAN实例
dbscan = DBSCAN(eps=0.2, min_samples=10)

# 进行密度聚类
y_pred = dbscan.fit_predict(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title('DBSCAN Clustering')
plt.show()