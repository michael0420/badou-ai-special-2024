"""
使用PCA求样本矩阵X的K阶降维矩阵
"""

import numpy as np


class CPCA(object):
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []

        self.centrX = self._centrX()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()
        print('X降维后是：', self.Z)

    # 矩阵中心化
    def _centrX(self):
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX

    # 协方差矩阵
    def _cov(self):
        C = np.dot(self.centrX.T, self.centrX) / (np.shape(self.centrX)[0] - 1)
        return C

    # 协方差矩阵的特征向量矩阵的降维矩阵
    def _U(self):
        eig_vals, eig_vectors = np.linalg.eig(self.C)
        idx = np.argsort(-1*eig_vals)
        UT = [eig_vectors[:, idx[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    # X矩阵降维
    def _Z(self):
        return np.dot(self.X, self.U)

if __name__ == '__main__':
    # 10样本3特征的样本集, 行为样例，列为特征维度
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
    print('X原矩阵：', X)
    K = np.shape(X)[1] - 1
    pca = CPCA(X, K)