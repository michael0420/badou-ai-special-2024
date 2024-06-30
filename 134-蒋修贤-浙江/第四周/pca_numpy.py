#!/usr/bin/env python3
import numpy as np
##数据准备
x = np.array([[-1,2,66,-1],[-2,6,58,-1],[-3,8,45,-2],[1,9,36,1],[2,10,62,1],[3,5,83,2]])

class PCA():
	def __init__(self,n_components):
		self.n_components = n_components
		
	def fit_transform(self,x):
		self.n_features_ = x.shape[1]
		
		x = x - x.mean(axis=0)
		self.convariance = np.dot(x.T,x)/x.shape[0]
		
		eig_vals,eig_vectors = np.linalg.eig(self.convariance)
		
		idx = np.argsort(-eig_vals)
		
		self.components_ = eig_vectors[:,idx[:self.n_components]]
		
		return np.dot(x,self.components_)
	
pca = PCA(n_components=2)
newX = pca.fit_transform(x)
print(newX)