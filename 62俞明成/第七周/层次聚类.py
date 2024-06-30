from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("TkAgg")

# 五个二维观测向量的数组（坐标矩阵）
X = np.array([[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]])
# linkage层次聚类函数
Z = linkage(X, "ward")
# fcluster根据层次聚类的结果（通常是linkage函数的输出）来划分样本到不同的簇中
f = fcluster(Z, 4, "distance")
fig = plt.figure(figsize=(5, 3))
# dendrogram绘制层次聚类的树状图
dn = dendrogram(Z)
plt.show()
