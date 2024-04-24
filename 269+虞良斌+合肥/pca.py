import numpy as np


class CPCA(object):

    def __init__(self, X, K):
        self.X = X  # 输入矩阵
        self.K = K  # 输出矩阵
        self.centreX = []  # X的中心化矩阵
        self.C = []  # 协方差矩阵
        self.U = []  # 降维转换矩阵
        self.Z = []  # 降维矩阵

        self.centreX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    # 求中心化矩阵
    def _centralized(self):
        centreX = []
        # 求输入矩阵每种特征的均值
        mean = np.array([np.mean(trait) for trait in self.X.T])
        # 中心化
        centreX = self.X - mean
        return centreX

    # 求协方差矩阵
    def _cov(self):
        # 中心化矩阵的元素个数
        m = np.size(self.centreX)
        C = np.dot(self.centreX.T, self.centreX) / m
        return C

    # 求降维转换矩阵
    def _U(self):
        # 求矩阵的特征值a、特征向量b
        a, b = np.linalg.eig(self.C)
        # 给出特征值降序的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        return Z

if __name__ == '__main__':
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
    pca = CPCA(X, K)
    print('样本矩阵X的降维矩阵Z:\n', pca.Z)