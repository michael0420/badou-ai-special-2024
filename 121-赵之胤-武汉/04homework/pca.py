import numpy as np

class Pca(object):
    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.centerX = []
        self.A = []
        self.UT = []
        self.U = []

        self.centerX = self._center()
        self.A = self._a()
        self.U = self._u()
        self.Z = self._z()


    def _center(self):
        mean = np.array([np.mean(i)for i in self.x.T])
        centerX = self.x - mean
        return centerX

    def _a(self):
        n = np.shape(self.centerX)[0]
        A = np.dot(self.centerX.T, self.centerX)/(n-1)
        return A

    def _u(self):
        a,b = np.linalg.eig(self.A)

        ind = np.argsort(-1 * a)

        UT = [b[:, ind[i]] for i in range(self.k)]
        U = np.transpose(UT)
        return U

    def _z(self):
        Z = np.dot(self.x, self.U)
        print('X shape:', np.shape(self.x))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
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
    pca = Pca(X, K)
