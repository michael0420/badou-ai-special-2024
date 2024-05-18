"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/4/20 12:49
"""
import numpy as np
from sklearn.decomposition import PCA

X = [[8, 4],
     [1, 2],
     [6, 12]]
X = np.array(X)
# 实例化pca, 参数降到一维
pca = PCA(n_components=1)
# 执行
new_X = pca.fit_transform(X)
print(X)
print(new_X)
