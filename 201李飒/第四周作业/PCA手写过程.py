""" 使用PCA 求 矩阵X 的 k阶 降维 矩阵Z"""

import numpy as np

class CPCA():

    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []
        self.C = []
        self.U = []
        self.Z = []

        self.centrX = self._centralizad()
        self.C = self._cov()
        self.U =self._U()
        self.Z =self._Z()

    def _centralizad(self):
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centrX = self.X - mean
        return centrX
    def _cov(self):
        ns = np.shape(self.X)[0]
        print(ns)
        C = np.dot(self.centrX.T,self.centrX)/ns
        return C
    def _U(self):
        a,b = np.linalg.eig(self.C)
        ind = np.argsort(-1*a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U
    def _Z(self):
        Z = np.dot(self.X,self.U)
        print("X的降维矩阵Z:\n",Z)
        return Z

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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