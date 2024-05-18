import numpy as np


class PCA(object):

    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.certrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()
        print(self.Z)

    def _centralized(self):
        mean = np.array([np.mean(attr) for attr in self.X.T])
        return self.X - mean

    def _cov(self):
        ns = np.shape(self.X)[0]
        return np.dot(self.certrX.T, self.certrX) / (ns - 1)

    def _U(self):
        a, b = np.linalg.eig(self.C)
        ind = np.argsort(-1 * a)
        UT = [b[:, ind[i]] for i in range(self.K)]
        return np.transpose(UT)

    def _Z(self):
        return np.dot(self.X, self.U)


if __name__ == '__main__':
    X = np.array([
        [47, 8, 7],
        [11, 16, 32],
        [12, 5, 1],
        [5, 21, 15],
        [47, 47, 2],
        [8, 22, 44],
        [6, 14, 20],
        [7, 9, 3],
        [48, 2, 1],
        [37, 14, 23],
    ])
pca = PCA(X, 2)
