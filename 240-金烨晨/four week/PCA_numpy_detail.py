import numpy as np

class CPCA:
    def __init__(self,A,K):
        self.A = A  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrA = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrA = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=AU

    def _centralized(self):
        '''矩阵X的中心化'''
        centrA = []
        mean = np.array([np.mean(i) for i in self.A.T])     #样本集的特征均值
        centrA = self.A - mean      #样本集的中心化
        return centrA

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        num = np.shape(self.centrA)[0]
        C = np.dot(self.centrA.T, self.centrA)/(num-1)
        return C

    def _U(self):
        a,b = np.linalg.eig(self.C)
        ind = np.argsort(-a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U

    def _Z(self):
        Z = np.dot(self.A, self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    A = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(A)[1] - 1
    pca = CPCA(A,K)