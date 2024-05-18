

import numpy as np

class CPCA(object):
    def __init__(self, X, K):
        self.X = X
        self.K = K
        # X中心化
        self.centrX=  []
        # 协方差矩阵
        self.C = []
        #降维转换矩阵
        self.U = []
        # 降维矩阵
        self.Z = []

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX

    def _cov(self):
        ns = np.shape(self.centrX)[0]
        # print(ns)
        C = np.dot(self.centrX.T, self.centrX)/ns
        print(C)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)
        ind = np.argsort(-1*a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print(U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__=='__main__':
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)
