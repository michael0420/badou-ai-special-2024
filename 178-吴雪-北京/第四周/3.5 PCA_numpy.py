"""
仿sklearn.decomposition中的PCA
PCA主成分分析：
    1. 样本矩阵X中心化
    2. 求协方差矩阵
    3. 求协方差矩阵的特征值和特征向量
    4. 给出特征值降序（从大到小排序）的索引序列
    5. 构建K阶降维的降维转换矩阵
    6. 降维矩阵
"""
import numpy as np


class CPCA(object):
    def __init__(self, K):
        self.K = K

    def _matrix(self, X):
        centre = X - X.mean(axis=0)
        covariance = np.dot(centre.T, centre) / X.shape[0]
        eigenvalues, eigenvectors = np.linalg.eig(covariance)
        index = np.argsort(-eigenvalues)
        dimensionality = eigenvectors[:, index[:self.K]]
        Z = np.dot(centre, dimensionality)
        return Z


if __name__ == '__main__':
    X = np.array([[-1, 2, 66, -1],
                  [-2, 6, 58, -1],
                  [-3, 8, 45, -2],
                  [1, 9, 36, 1],
                  [2, 10, 62, 1],
                  [3, 5, 83, 2]])
    pca = CPCA(K=2)
    matrix = pca._matrix(X)
    print(matrix)
