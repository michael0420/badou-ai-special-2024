

import numpy as np
from sklearn.decomposition import PCA
# X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4

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
pca = PCA(n_components=2)   #降到2维
A = pca.fit(X)                  #执行
print(A)
newX=pca.fit_transform(X)   #降维后的数据
print(newX)                  #输出降维后的数据