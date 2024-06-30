# PCA - principle component analysis - transform data into a set of linearly uncorrelated features
# use for dimension reduction, noise reduction, feature extraction and data compression
# Pro: no need to preset arguments. Cons: cannot modify setting based on knowledge of the dataset

# Use PCA for dimension reduction (step by step without using build-in function)

import numpy as np

# define as a class for encapsulation, reusability and modularity
class CPCA(object):
    # 1- initialization - initial variables and define matrix X with m rows sample and n features, reduce to K dimension
    def __init__(self, X, K):
        self.X = X         # sample matrix X
        self.K = K         # target final dimension
        self.centrX = []   # centralized matrix
        self.C = []        # covariance matrix
        self.U = []        # transformation matrix
        self.Z = []        # reduced matrix

        self.centrX = self._centralized()  # "_" is a signal for internal use in the class only
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    # 2 - Centering the matrix
    # compute the mean value of each feature, and then variables subtract the means to center the matrix
    # shifts the mean of each feature to zero. centrX = X - mean
    def _centralized(self):
        print('the sample matrix: ', self.X)
        mean = np.array([np.mean(attr) for attr in self.X.T])  #mean of each feature(turned to rows by transpose)
        print('the mean of feature before centralize: ', mean)
        centrX = self.X - mean  # subtract the mean to center the matrix
        print('centralized matrix: ', centrX)
        return centrX

    # 3 - Covariance Matrix of the centralized matrix
    # covariance shows the linear relationship between two features
    # PCA aims to transform the dataset into a new dataset with features that captures the max. of variance
    # the new set of features are uncorrelated, and ordered by amount of variance
    def _cov(self):
        ns = np.shape(self.centrX)[0]  # get the total number of samples(features)
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1) # covariance matrix(ns-1 unbiased estimate, bessel's correction)
        print('covariance matrix: ', C)
        return C

    # 4 - Transformation Matrix
    def _U(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.C)
        ind = np.argsort(-1*eigenvalues) # Indices for sorting eigenvalues in descending order (-1)
        # compute the transformation matrix for K dimension
        UT = [eigenvectors[:, ind[i]] for i in range(self.K)]
        U = np. transpose(UT)
        print('%d dimension transformation matrix: '%self.K, U)
        return U

    # 5 - Dimension reduction (Z=XU)
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('Reduced matrix: ', Z)
        return Z


if __name__=='__main__':
    '10 samples with 3 features, rows as samples，col as features'
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
    print('Data set of 10 samples with 3 features:\n', X)
    pca = CPCA(X,K)

