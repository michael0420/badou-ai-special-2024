"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np

class CPCA(object):

    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centrX = []  #中心化
        self.C = []  #协方差矩阵C
        self.U = []  #降维转换矩阵
        self.Z = []  #降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])   #求均值
        centrX = self.X - mean  #中心化
        return centrX

    def _cov(self):
        '''求协方差矩阵C'''
        # 样本数
        ns = np.shape(self.centrX)[0]
        # 求协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        return C

    def _U(self):
        '''求降维转换矩阵U'''
        # 求特征值和特征向量
        a, b = np.linalg.eig(self.C)  # a为特征值，b为特征向量
        # 根据特征值倒序排序
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[7, 32, 7],
                  [88, 21, 0],
                  [12, 60, 31],
                  [2, 15, 46],
                  [42, 10, 60],
                  [14, 3, 60],
                  [91, 29, 5],
                  [69, 9, 3],
                  [20, 18, 39],
                  [18, 27, 49]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)
