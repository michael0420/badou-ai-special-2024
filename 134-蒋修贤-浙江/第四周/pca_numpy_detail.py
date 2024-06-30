#!/usr/bin/env python3
import numpy as np

class CPCA(object):
	
	def __init__(self,X,K):
		self.X = X
		self.K = K
		self.centerX = []
		self.C = []
		self.U = []
		self.Z = []
		##矩阵中心化
		
		self.centerX = self._centerlized()
		self.C = self._cov()
		self.U = self._u()
		self.Z = self._z()
		
	
	def _centerlized(self):
		centerX = []
		mean = np.array([np.mean(attr) for attr in self.X.T])
		centerX = self.X - mean
		return centerX
	
	def _cov(self):
		ns = np.shape(self.centerX)[0]
		C = np.dot(self.centerX.T,self.centerX)/(ns-1)
		return C
	
	def _u(self):
		a,b = np.linalg.eig(self.C)
		ind = np.argsort(-1*a)
		UT = [b[:,ind[i]] for i in range(self.K)]
		return np.transpose(UT)
	
	def _z(self):
		return np.dot(self.X,self.U)

X = np.array([[10,15,29],[15,46,13],[23,21,30],[11,9,35],[42,45,11],[9,48,5],[11,21,14],[8,5,15],[11,12,21],[21,20,25]])
K = np.shape(X)[1]-1
pca = CPCA(X,K)
print(pca)
	