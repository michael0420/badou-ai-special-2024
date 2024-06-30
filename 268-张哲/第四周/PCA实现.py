# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""
import numpy as np
from sklearn.decomposition import PCA
class CPCA(object):
    def __init__(self,K):
        self.K = K #降阶目标矩阵的维度
    def fit_transform(self,X):
        #求去中心化的协方差矩阵
        X = X - X.mean(axis=0)
        #注：去中心化后的原始矩阵，其转置矩阵和本身相乘，并除以特征的样本个数，其值拆开后和协方差矩阵等同
        self.convariance = np.dot(X.T,X)/X.shape[0]
        # 求协方差矩阵的特征值和特征向量
        a,b = np.linalg.eig(self.convariance)
        # 给出特征值降序的topK的索引序列
        idx = np.argsort(-1 * a)
        #降维特征矩阵
        self.components = b[:,idx[:self.K]]
        #对原始矩阵进行降维处理
        return np.dot(X,self.components)

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca_ = CPCA(K)
    new_X = pca_.fit_transform(X)
    print(new_X)
    #串口实现PCA
    pca = PCA(n_components=2)#降到2维
    pca.fit(X)                  #执行
    pca_X=pca.fit_transform(X)  #降维后的数据
    print(pca_X)
