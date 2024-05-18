'''
使用PCA求样本矩阵X的K阶降维矩阵Z
'''

import numpy as np

class CPCA(object):

    def __init__(self,X,k):
        '''
        :param X:样本矩阵X
        :param k: X的降维矩阵的阶数
        '''
        self.X = X
        self.k = k
        self.centrX = self._centralized() #矩阵的中心化
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        '''求样本矩阵X的中心化矩阵centrX'''
        print('样本矩阵X:\n',self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print('特征均值:\n',mean)
        centrX = self.X - mean
        print('X的中心化centrX:\n',centrX)
        return centrX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T,self.centrX)/(ns - 1)
        print('样本矩阵X的协方差C:\n',C)
        return C

    def _U(self):
        '''求X的降维转换矩阵U，shape=(n,k)，n是X的特征维度总数，k是降维矩阵的特征维度'''
        a,b = np.linalg.eig(self.C) #求特征值和特征向量
        print('协方差矩阵C的特征值：\n',a)
        print('协方差矩阵C的特征向量：\n',b)
        ind = np.argsort(-1*a) #特征值降序的topK的索引序列
        UT = [b[:,ind[i]] for i in range(self.k)]
        U = np.transpose(UT) #构建K阶降维的降维转换矩阵U
        print('%d阶降维转换矩阵U:\n'%self.k,U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z，shape=(m,k)，n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X,self.U)
        print('X shape:',np.shape(self.X))
        print('U shape:',np.shape(self.U))
        print('Z shape:',np.shape(Z))
        print('样本矩阵X的降维矩阵Z：\n',Z)
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
    k = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    print(k)
    pca = CPCA(X, k)





