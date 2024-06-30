"""
用代码直译PCA主成分分析（手工实现）
使用PCA求样本矩阵X的K阶降维矩阵Z

1. 定义各种参数
    样本矩阵X
    K阶降维矩阵的K值
    矩阵X的中心化centre-centralized
    协方差矩阵C-covariance
    降维转换矩阵D-dimensionality
    降维矩阵Z-matrix
2. 样本矩阵X的零均值化（中心化）centralized
3. 样本矩阵X(零均值化后)的协方差矩阵 covariance
4. np.linalg.eig()对协方差矩阵求特征值和特征向量，构建K阶降维的降维转换矩阵 dimensionality
5. 样本矩阵X的降维矩阵Z=XD matrix
"""
import numpy as np


class CPCA(object):
    # 1. 定义各种参数
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centre = []
        self.C = []
        self.D = []
        self.Z = []
        self.centre = self._centralized()
        self.C = self._covariance()
        self.D = self._dimensionality()
        self.Z = self._matrix()

    # 2. 样本矩阵X的零均值化（中心化）centralized
    def _centralized(self):
        centre = self.X - self.X.mean(axis=0)
        return centre

    # 3. 样本矩阵X(零均值化后)的协方差矩阵 covariance
    def _covariance(self):
        C = np.dot(self.centre.T, self.centre) / self.X.shape[0]
        return C

    # 4. np.linalg.eig()对协方差矩阵求特征值和特征向量，构建K阶降维的降维转换矩阵 dimensionality
    def _dimensionality(self):
        a, b = np.linalg.eig(self.C)
        index = np.argsort(-a)
        D = b[:, index[:self.K]]
        return D

    # 5. 样本矩阵X的降维矩阵Z=XD matrix
    def _matrix(self):
        Z = np.dot(self.centre, self.D)
        print('X shape:', self.X.shape)
        print('D shape:', self.D.shape)
        print('Z shape:', Z.shape)
        return Z


if __name__ == '__main__':
    # 10样本3特征的样本集，行为样例，列为特征维度
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
    K = X.shape[1] - 1
    print(K)
    pca = CPCA(X, K)
    print("样本矩阵X的降维矩阵Z:\n", pca.Z)
