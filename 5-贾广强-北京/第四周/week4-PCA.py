#!/usr/bin/env python
# coding: utf-8

# In[3]:


#使用nmpy 实现主成分分析
import numpy as np


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]  #获取特征数量
        X = X - X.mean(axis=0)  #按列方向   数据中心化
        self.covariance = np.dot(X.T, X) / X.shape[0]  # 求协方差矩阵
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)  #求特征值及特征向量
        idx = np.argsort(-eig_vals)
        self.components_ = eig_vectors[:, idx[:self.n_components]]  #获取降维后的矩阵
        return np.dot(X, self.components_)  #返回投影后矩阵


pca = PCA(n_components=2)
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1],
              [2, 10, 62, 1], [3, 5, 83, 2]])
newX = pca.fit_transform(X)
print(newX)


# In[9]:


#使用sklearn 实现PCA
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1],
              [2, 10, 62, 1], [3, 5, 83, 2]])
pca = PCA(n_components=2)
pca.fit(X)
newX = pca.fit_transform(X)
print(newX)

