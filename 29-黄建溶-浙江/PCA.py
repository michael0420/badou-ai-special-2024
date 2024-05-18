import numpy as np
class CPCA (object):
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean                         #样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):                                      # 求样本矩阵X的协方差矩阵C
        ns = np.shape(self.centrX)[0]                      # 取出样本集的样例总数
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  # 计算样本矩阵的协方差矩阵C
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)                 # 特征值赋值给a，对应特征向量赋值给b
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1 * a)                     # 进行降序排列
        UT=[b[:,ind[i]] for i in range(self.K)]      # 对排序好的特征值进行取值
        U = np.transpose(UT)                         # 转置
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
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
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca= CPCA(X,K)
