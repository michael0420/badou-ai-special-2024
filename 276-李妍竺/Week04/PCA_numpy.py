import numpy as np

class PCA():
    def __init__(self,n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        X = X - X.mean(axis=0)   # axis=0,按列求（在输出中按照行来求，得到的是矩阵的列）
        self.covariance = np.dot(X.T,X)/X.shape[0]

        eig_values,eig_vectors = np.linalg.eig(self.covariance)

        index = np.argsort(-eig_values)
        self.components = eig_vectors[:,index[:self.n_components]]
        # [:,i]:输出每个数组或矩阵的第i+1个元素组成新的数组或矩阵输出

        return np.dot(X,self.components)



X = np.array([[14,16,31],
                  [16,34,88],
                  [22,33,44],
                  [18,25,17],
                  [28,4,31],
                  [2,44,55],
                  [31,35,13],
                  [24,6,15],
                  [27,25,26],
                  [19,12,11]])
pca = PCA(n_components = X.shape[1]-1)
newX = pca.fit_transform(X)
print(newX)