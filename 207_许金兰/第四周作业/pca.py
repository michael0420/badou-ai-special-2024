"""
@author: 207-xujinlan
numpy实现pca
"""

import numpy as np


def pca_np(X, k):
    """
    使用numpy实现pca
    :param X: 样本数据，m*n矩阵
    :param k: 降维后的列数目
    :return:
    """
    mu = np.array([np.mean(i) for i in (X.T)])  # 求每列的平均值
    centerx = X - mu  # 去中心化
    cov_x = np.dot(centerx.T, centerx)/X.shape[0]  # 求协方差矩阵
    a, v = np.linalg.eig(cov_x)  # 求协方差矩阵的特征值和特征向量
    ind = np.argsort(-a)  # 特征值降序排序
    v_k = [v[:, ind[i]] for i in range(k)]  # 取前k个特征值对应的特征向量
    v_x = np.transpose(v_k)  # 特征向量矩阵转换
    U = np.dot(X, v_x)  # 样本矩阵与特征向量矩阵相乘，实现降到K维
    return U


if __name__ == '__main__':
    X = np.arange(12).reshape([3, 4])
    X_pca = pca_np(X, 2)
