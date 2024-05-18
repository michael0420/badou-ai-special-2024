#pca 主成分分析
'''
API方式
'''

import numpy as np
import sklearn.decomposition as dp

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

pca = dp.PCA(n_components=2)
newX = pca.fit_transform(X)

print(newX)




