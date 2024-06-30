"""
@author: huangyunxuan

PCA

"""

import numpy as np


#
class PCA():
    def __init__(self,n_components):
        self.n_components = n_components
    def fit_transform(self,x):
        self.n_feature_ = x.shape[1]
        x = x - x.mean(axis = 0)
        self.covariance = np.dot(x.T,x)/x.shape[0]
        eig_vals,eig_vectors = np.linalg.eig(self.covariance)
        print("eig_vals=", eig_vals)
        print("eig_vectors=",eig_vectors)
        idx = np.argsort(-eig_vals)
        print("idx=",idx)
        self.components_ = eig_vectors[:,idx[:self.n_components]]
        return np.dot(x,self.components_)
pca = PCA(n_components=2)
x = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
newX=pca.fit_transform(x)
print(newX)
print("X:", x)
print("X.mean:", x.mean(axis=1))
# 调用numpy 手搓







# 调用sklearn.decomposition PCA接口
# pca = PCA(n_components = 2)
# X = np.array([[-1,2,66,-1],
#               [-2,6,58,-1],
#               [-3,8,45,-2],
#               [1,9,36,1],
#               [2,10,62,1],
#               [3,5,83,2]])
# print(pca.fit_transform(X))