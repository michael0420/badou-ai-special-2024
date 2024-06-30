'''
调用sklearn接口进行主成分分析
@zeno wang
'''

import numpy as np
from sklearn.decomposition import PCA  # conda install scikit-learn
X = np.array([[-1, 2, 66, -1],
              [-2, 6, 58, -1],
              [-3, 8, 45, -2],
              [1, 9, 36, 1],
              [2, 10, 62, 1],
              [3, 5, 83, 2]])
pca = PCA(n_components=2)  # 降维阶数设置2
pca.fit(X)  # 执行
X_new = pca.fit_transform(X)  # 降维后的数据
print(X_new)
