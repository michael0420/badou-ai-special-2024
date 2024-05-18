# coding=utf-8

import numpy as np
from sklearn.decomposition import PCA

X = np.array(
    [[-2, 23, 86, -11], [-9, 5, 98, 1], [3, 18, 23, 1], [-1, 19, 26, 10], [12, 100, 162, 3], [13, 45, 283, 29]])  # 导入数据，维度为4
pca = PCA(n_components=2)  # 降到2维
pca.fit(X)  # 执行
newX = pca.fit_transform(X)  # 降维后的数据
print(newX)  # 输出降维后的数据
