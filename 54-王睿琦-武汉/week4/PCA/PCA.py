# -*- coding: utf-8 -*-
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        # n指的是降到n维

    def analysis(self, x):
        # 获取数据集的特征数
        self.n_features = x.shape[1]
        # 对数据进行中心化，即减去每列的均值
        x = x - x.mean(axis=0)  # axis=0 返回每一列的均值，axis=1 返回每一行的均值
        # 计算协方差矩阵
        self.covariance = np.dot(x.T, x) / x.shape[0]
        # 特征值和特征向量
        # 特征向量矩阵eig_vectors中的每一列都对应着特征值数组eig_vals中相应位置的特征值。
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        print('样本集的特征值:\n', eig_vals)
        print('样本集的特征向量:\n', eig_vectors)
        # 降序特征值 返回的是对应的索引排序
        # eg: [5,9,2] => [1,0,2]
        ids = np.argsort(-eig_vals)
        # 降维矩阵
        self.components = eig_vectors[:, ids[:self.n_components]]
        # 映射
        return np.dot(x, self.components)

# 设置降到3维
pca = PCA(n_components=3)
X = np.array([[-1, 2, 66, -1, 5], 
              [12, -2, 6, 58, -1], 
              [11, -3, 8, 45, -2], 
              [5, 1, 9, 36, 1], 
              [6, 2, 10, 62, 1], 
              [5, 3, 5, 83, 2]])  # 导入数据，维度为5
newX = pca.analysis(X)
print("降维后的数据:", newX)  # 输出降维后的数据
