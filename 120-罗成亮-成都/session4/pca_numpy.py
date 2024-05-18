import numpy as np


class PCA(object):

    def __init__(self, K):
        self.K = K

    def fit_transform(self, X):
        X = X - X.mean(axis=0)

        cov = np.dot(X.T, X) / (X.shape[0] - 1)

        a, b = np.linalg.eig(cov)

        ind = np.argsort(-a)

        UT = [b[:, ind[i]] for i in range(self.K)]
        # 对X进行降维
        return np.dot(X, np.transpose(UT))


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
    pca = PCA(2)
    result = pca.fit_transform(X)
    print(result)