#!/usr/bin/env python
# coding: utf-8

# In[34]:


'''
过程：
1.获取目标函数self，为m*n阶，以及要降维到k，最后的矩阵是q
2.均值中心化：求得方差
3.获取中心化后的协方差矩阵matrix：
4.求得协方差函数的特征值lamda和对应的各个特征矩阵
5.取前k值，得到相应的矩阵
6.输出2 3 4 5
'''

import numpy as np

class PCA():

    def __init__ (self, X, k):
        
        #self为目标矩阵。降低的维度为k，m为降低维度后的函数
        
        
        self.X = np.array(X) #目标矩阵
        self.k = k
        self.X_centralized = []
        self.cov = [] #协方差矩阵
        self.Y = [] #k维度矩阵
        
        self.X_centralized = self.Centralized() 
        self.cov = self.PCAmatrix() 
        self.U = self.Dematrix()
        self.Y = self.New()
        
        print ("样本集:\n", self.X)
        print ("样本去中心化:\n", self.X_centralized)
        print ("协方差矩阵:\n", self.cov)
        print("降到 %d 维后的目标矩阵:\n" % self.k, self.Y)      

    def Centralized (self):
        X_centralized = []
        Mean = np.mean(self.X, axis=0) #对列求中心化
        print ("样本均值:\n", Mean)
        X_centralized = np.subtract(self.X, Mean)
        return X_centralized
        
    def PCAmatrix (self):
        m = np.shape(self.X_centralized)[0] #数量
        cov = np.dot(self.X_centralized.T, self.X_centralized)/(m-1) #求出协方差矩阵
        return cov
    
    def Dematrix (self):
        i, h = np.linalg.eig(self.cov) #np自带计算协方差矩阵的特征值和特征向量
        #对i进行就降序排列
        #选取前k值个特征向量
        U = h[:, -self.k:]
        return U

    def New (self):
        #将矩阵投影到选取的特征向量中得到新矩阵
        Y = np.dot(self.X_centralized, self.U)
        return Y

if __name__ == '__main__':
    X = np.array([[3,5,7], [2,8,9], [4,7,3], [7,2,4], [2,9,1],[4,7,2]])  
    k = np.shape(X)[1] - 1
    
    newX = PCA(X, k)  

