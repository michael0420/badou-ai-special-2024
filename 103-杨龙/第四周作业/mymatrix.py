# -*- coding:utf-8 -*-

__author__ = "Young"

import numpy as np
from sklearn.decomposition import PCA


class MyMatrix(object):
    """
    用PCA求样本矩阵X的降维矩阵Z，k为Z的维度(阶数)。
    """

    def __init__(self, X, k):
        """
        X为样本矩阵，k为PCA降维后的矩阵阶数
        :param X:
        :param k:
        """
        self.X = X  # 样本矩阵
        self.k = k  # PCA后的Matrix阶数
        self.center_X = []  # 中心化后的X
        self.covariance_X = []  # X协方差(covariance)矩阵
        self.transposition_Z = []  # 降维后的矩阵的转换矩阵
        self.Z = []  # 降维后的矩阵

    def principal_component_analysis(self):
        """PCA"""
        self.center_X = self.__centralize()
        self.covariance_X = self.__covariance()
        self.transposition_Z = self.__transposed_matrix()
        self.Z = self.__pca_matrix()

    def __centralize(self):
        """
        样本矩阵X的中心化
        :return:
        """
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        return self.X - mean  # 样本集去中心化：样本集 - 均值矩阵

    def __covariance(self):
        """求样本集的协方差矩阵"""
        sample_total_count = np.shape(self.center_X)[0]
        return np.dot(self.center_X.T, self.center_X) / (sample_total_count - 1)

    def __transposed_matrix(self):
        """求X的降维转换矩阵U，shape = (n,k)。n是X的维度数，k是Z的维度数"""
        # 求协方差矩阵的特征值和特征向量，函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        eigen_value, eigen_vector = np.linalg.eig(self.covariance_X)
        eigen_value_order_index = np.argsort(-1 * eigen_value)  # 得到特征值降序的top k的索引序列，降维到k阶，只需要取k种特征即可
        ordered_matrix = [eigen_vector[:, eigen_value_order_index[i]] for i in range(self.k)]  # 取k个特征向量组成矩阵
        return np.transpose(ordered_matrix)  # 转成目标矩阵

    def __pca_matrix(self):
        """按照Z=XU求降维矩阵Z，shape = (m,k)"""
        return np.dot(self.center_X, self.transposition_Z)

    def pca_numpy(self):
        """
        演示调用numpy的函数进行PCA
        :return:
        """
        self.center_X = self.X - self.X.mean(axis=0)  # 中心化
        # 协方差矩阵
        self.covariance_X = np.dot(self.center_X.T, self.center_X) / self.center_X.shape[0]
        eig_values, eig_vectors = np.linalg.eig(self.covariance_X)
        idx = np.argsort(-eig_values)  # 降序的特征值序号
        self.transposition_Z = eig_vectors[:, idx[:self.k]]
        self.Z = np.dot(self.center_X, self.transposition_Z)

    def pca_sklearn(self):
        """
        演示调用sklearn库方法进行PCA
        :return:
        """
        x = self.X
        pca = PCA(self.k)
        pca.fit(x)
        self.Z = pca.fit_transform(x)
