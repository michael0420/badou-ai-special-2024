#!/usr/bin/env python
# coding: utf-8

# In[14]:


# detail版
import numpy as np
class CPCA(object):
    def __init__(self,X,K):
        self.X =X
        self.K =K
        self.centrX =[]
        self.C=[]
        self.U =[]
        self.Z =[]
        
        self.centrX =self._centrX()
        self.C =self._C()
        self.U =self._U()
        self.Z =self._Z()
        
    def _centrX(self):
        centrX =[]
        mean =np.array([np.mean(attr) for attr in self.X.T])
        centrX =X-mean
        return centrX
    def _C(self):
        ns =X.shape[0]
        C =np.dot(self.centrX.T,self.centrX)/(ns-1)
        return C
    def _U(self):
        a,b =np.linalg.eig(self.C)
        ind =np.argsort(-1*a)
        UT =[b[:,ind[i]] for i in range(self.K)]
        U =np.transpose(UT)
        return U
    def _Z(self):
        Z =np.dot(self.X,self.U)
        print(Z)
        return Z
if __name__ =="__main__":
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
    
    K =np.shape(X)[1]-1
    pca=CPCA(X,K)
    


# In[18]:


# PCA简化版
import numpy as np
class PCA():
    def __init__(self,n_components):
        self.n_components =n_components
    def fit_transform(self,X):
        nums =X.shape[0]
            #计算均值和协方差
        X =X-X.mean(axis=0)
        self.covariance =np.dot(X.T,X)/nums
            #计算特征值和特征向量
        eig_vals,eig_vectors =np.linalg.eig(self.covariance)
        idx =np.argsort(-eig_vals)
        self_components =eig_vectors[:,idx[:self.n_components]]
            
            #降维后特征矩阵
        return np.dot(X,self_components)
        

pca =PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]]) 
newX =pca.fit_transform(X)
print(newX)


# In[19]:


# PCA接口版
import numpy as np
from sklearn.decomposition import PCA #调用PCA接口
X =np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  
pca =PCA(n_components=2)
newX =pca.fit_transform(X)
print(newX)

