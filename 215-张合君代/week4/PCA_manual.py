# -*- coding: utf-8 -*-
"""
@author: zhjd
手写PCA（利用numpy做数组、矩阵运算）
"""

import numpy as np


class PCA(object):
    """
    手写PCA（利用numpy做数组、矩阵运算）
    """

    def __init__(self, X, K):  # self是什么？
        """
        :param X,原始特征样本集
        :param K,保留维度数量，即新矩阵列数
        """
        self.X = X
        self.K = K
        self.centreX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 原始特征样本集X的降维转换矩阵
        self.Z = []  # 原始特征样本集X的降维矩阵Z
        print('原始特征样本集X:\n', self.X)
        self.centreX = self._centralized()
        print('原始特征样本集X的中心化后:\n', self.centreX)
        self.C = self._covariance()
        self.U = self._eigenvector_matrix()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        return self.X - mean

    def _covariance(self):
        """
        原始特征样本集X的协方差矩阵
        """
        # 样本集的样例总数
        ns = np.shape(self.centreX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centreX.T, self.centreX) / (ns - 1)
        print('原始特征样本集X的协方差矩阵C:\n', C)
        return C

    def _eigenvector_matrix(self):
        """
        求X的k维度的特征向量矩阵, shape=(n,k), n是X矩阵的原维度数量，k是降维矩阵的维度数量
        """
        a, b = np.linalg.eig(
            self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)

        return U

    def _Z(self):
        """按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数"""
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('原始特征样本集X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    """
    10样本3特征的样本集, 行为样例，列为特征维度
    """
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
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X, K)
