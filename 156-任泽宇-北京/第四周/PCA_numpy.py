

import numpy as np

class PCA():
    def __init__(self, K):
        self.K = K

    def fit_transform(self,X):
        self.n_features_ = X.shape[1]
        X1 = X - X.mean(axis=0)
        self._cov = np.dot(X1.T,X1)/X1.shape[0]
        a,b = np.linalg.eig(self._cov)
        ind = np.argsort(-1*a)
        self.components_ = b[:,ind[:self.K]]
        # print(X)
        # print(self.components_)
        return np.dot(X,self.components_)

pca = PCA(K=2)
X = np.array([[10, 15, 29],[15, 46, 13],[23, 21, 30],[11, 9,  35],[42, 45, 11], [9,  48, 5],[11, 21, 14], [8,  5,  15],[11, 12, 21], [21, 20, 25]])
newX=pca.fit_transform(X)
print(newX)