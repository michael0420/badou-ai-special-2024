import numpy as np

class PCA():
    def __init__(self, n_componets):
        self.n_componets = n_componets
    
    def fit_transform(self, X):
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        eig_vals, eig_ventors = np.linalg.eig(self.covariance)
        idx = np.argsort(-eig_vals)
        self.componests_ = eig_ventors[:, idx[:self.n_componets]]
        
        return np.dot(X, self.componests_)


if __name__ == "__main__":
    pca = PCA(n_componets=2)
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
    newX = pca.fit_transform(X)
    print(newX)