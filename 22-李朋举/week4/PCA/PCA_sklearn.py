# coding=utf-8

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
pca = PCA(n_components=2)  # 降到2维  创建了一个主成分分析对象pca，并指定了要保留的主成分数量为 2。
pca.fit(X)  # 执行 使用数据矩阵X对pca对象进行拟合，也就是训练pca对象
newX = pca.fit_transform(X)  # 降维后的数据  使用拟合后的pca对象对数据矩阵X进行降维处理，将维度从 4 降到 2，并将结果存储在新的矩阵newX中。
print(newX)  # 输出降维后的数据
