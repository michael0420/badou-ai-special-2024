import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def k_means_cluster(X, k, max_iter=300, tol=1e-4):
    kmeans = KMeans(n_clusters=k, max_iter=max_iter, tol=tol)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_


def evaluate_cluster(X, labels, centers):
    # 计算平均距离
    avg_dist = np.mean([np.linalg.norm(X[i] - centers[labels[i]]) for i in range(len(X))])
    # 计算 silhouette_score
    silhouette_avg = silhouette_score(X, labels)
    print("平均距离: ", avg_dist)
    print("平均 silhouette_score: ", silhouette_avg)


# 示例数据集
data = np.random.rand(100, 2)  # 生成100个二维数据点



k = 2  # 设定聚类数目为2


# print (data)

# 执行k-means算法
labels, centers = k_means_cluster(data, k)

# 评估聚类效果
evaluate_cluster(data, labels, centers)



########################################




clf = KMeans(n_clusters=2)
y_pred = clf.fit_predict(data)

# 输出完整Kmeans函数，包括很多省略参数
print(clf)
# 输出聚类预测结果
print("y_pred = ", y_pred)

"""
可视化绘图
"""

import numpy as np
import matplotlib.pyplot as plt

# 获取数据集的第一列和第二列数据 使用for循环获取 n[0]表示X第一列
x = [n[0] for n in data]
print(x)
y = [n[1] for n in data]
print(y)

''' 
绘制散点图 
参数：x横轴; y纵轴; c=y_pred聚类预测结果; marker类型:o表示圆点,*表示星型,x表示点;
'''
plt.scatter(x, y, c=y_pred, marker='x')

# 绘制标题
plt.title("Kmeans-Basketball Data")

# 绘制x轴和y轴坐标
plt.xlabel("assists_per_minute")
plt.ylabel("points_per_minute")

# 设置右上角图例
plt.legend(["A", "B", "C"])

# 显示图形
plt.show()











