# PCA via numpy package only

import numpy as np


class PCA():
    # 1 - initialization
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        # get the total features
        self.n_features = X.shape[1]
        # 2 - centralize
        X = X - X.mean(axis=0)
        # 3 - covariance matrix
        self.covariance = np.dot(X.T,X)/X.shape[0]
        # 4 - transformation matrix from eigenvalue and eigenvector
        eig_vals, eig_vecs = np.linalg.eig(self.covariance)
        idx = np.argsort(-eig_vals) # rank then in decreasing order
        self.components_ = eig_vecs[:,idx[:self.n_components]] # transformation matrix
        # 5 - reduce dimension
        return np.dot(X, self.components_)


# apply the class PCA
pca = PCA(n_components=2)  # reduced to 2 dimensions
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  # 4 dimensions
newX = pca.fit_transform(X)
print(newX)
