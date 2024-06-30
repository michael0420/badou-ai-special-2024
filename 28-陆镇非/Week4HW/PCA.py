# Author: Zhenfei Lu
# Created Date: 4/15/2024
# Version: 1.0
# Email contact: luzhenfei_2017@163.com, zhenfeil@usc.edu

import numpy as np

class PCA(object):
    def __init__(self, features, targetDimension):
        self.features = features
        self.targetDimension = targetDimension
        self.N = self.features.shape[0]


    def decentralize(self) -> None:
        self.decentralFeatures = np.zeros((self.features.shape[0], self.features.shape[1]), self.features.dtype)
        self.decentralFeatures = self.features - np.mean(self.features, axis=0)
        return

    def getCovariance(self) -> None:
        self.covariance = (1/(self.N-1)) * (self.decentralFeatures.T @ self.decentralFeatures)
        return

    def getEigenVectors(self) -> np.ndarray:
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        idx = np.argsort(-eig_vals)
        new_eig_vectors = np.zeros((eig_vectors.shape[0], self.targetDimension), eig_vectors.dtype)
        for i in range(0, self.targetDimension):
            new_eig_vectors[:, i] = eig_vectors[:, idx[i,]]
        return new_eig_vectors

    def reduceFeatureDim(self) -> np.ndarray:
        self.decentralize()
        self.getCovariance()
        w = self.getEigenVectors()
        return self.features @ w