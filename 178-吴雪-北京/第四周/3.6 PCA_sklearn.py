"""
PCA是无监督学习算法
sklearn.decomposition
"""
import numpy as np
from sklearn.decomposition import PCA


X = np.array([[-1, 2, 66, -1],
              [-2, 6, 58, -1],
              [-3, 8, 45, -2],
              [1, 9, 36, 1],
              [2, 10, 62, 1],
              [3, 5, 83, 2]])
pca = PCA(n_components=2)
matrix = pca.fit_transform(X)
print(matrix)
