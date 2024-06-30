
import numpy as np


class SPCA(object):
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, X, K):
        '''
        :param X,样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.W = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()   #中心化
        self.C = self._cov()  # 协方差矩阵
        self.W = self._W()
        self.Z = self._Z()  # Z=XW求得

    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  #样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    '''
    求样本矩阵X的协方差矩阵C
    '''
    def _cov(self):
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)# 中心矩阵的转置*样本个数分之1
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    '''
    求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
    '''
    def _W(self):
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        WT = [b[:, ind[i]] for i in range(self.K)]
        W = np.transpose(WT) # 转置
        print('%d阶降维转换矩阵U:\n' % self.K, W)
        return W

    '''
    按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数
    '''
    def _Z(self):
        Z = np.dot(self.X, self.W)
        print('X shape:', np.shape(self.X))
        print('W shape:', np.shape(self.W))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    # 手动计算PCA步骤
    # 1.中心化 2.求协方差矩阵  3.特征值 特征矩阵
    X = np.array([[6, 25, 18],
                  [16, 33, 55],
                  [26, 43, 24],
                  [15, 9, 46],
                  [52, 41, 21],
                  [9, 48, 5],
                  [8, 9, 13],
                  [81, 56, 55],
                  [13, 31, 45],
                  [40, 41, 23]])
    K = 2     #  降维后维度
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = SPCA(X, K)
