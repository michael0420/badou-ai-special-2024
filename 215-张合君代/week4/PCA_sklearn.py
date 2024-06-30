# -*- coding: utf-8 -*-
"""
@author: zhjd
使用sklearn接口实现
"""

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
pca = PCA(n_components=2)  # 降到2维
pca.fit(X)  # 执行
newX = pca.fit_transform(X)  # 降维后的数据
print(newX)  # 输出降维后的数据
