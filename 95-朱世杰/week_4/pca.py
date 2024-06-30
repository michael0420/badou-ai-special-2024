import numpy as np

class pca(object):

    def __init__(self, X, K):
        '''
        :param X,样本矩阵X
        :param K,降到k维
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized() # 中心化
        self.C = self._cov() # 协方差矩阵
        self.U = self._U() # 降维转换矩阵
        self.Z = self._Z()  # 降维后的矩阵

    def _centralized(self):
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值，X.T装置后每行都是一种特征值的列表，遍历并求平均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  ##样本集的中心化
        print('样本矩阵X的中心化矩阵centrX:\n', centrX)
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本数
        m = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (m - 1)  # D = 1/(m-1) * Z.T * Z
        print('样本矩阵X的中心化协方差矩阵C:\n', C)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)  # 求协方差矩阵的特征值a，特征向量b
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 特征值降序
        ind = np.argsort(-1 * a)
        # 对应前k位的特征向量矩阵
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        # 原始矩阵与选取后的特征向量矩阵乘积
        Z = np.dot(self.X, self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    pca = pca(X, K)
