import numpy as np


def center_covariance_matrix(X):
    """
    计算中心化协方差矩阵

    参数:
    X : numpy array of shape (n_samples, n_features)
    输入数据矩阵

    返回:
    covariance matrix : numpy array of shape (n_features, n_features)
    中心化协方差矩阵
    """
    # 计算平均值
    mean_vector = np.mean(X, axis=0)

    # 中心化数据
    X_centered = X - mean_vector

    # 计算协方差矩阵
    covariance_matrix = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

    return covariance_matrix


# 示例使用
data = np.array([[1, 2], [3, 4], [5, 6]])
centered_cov = center_covariance_matrix(data)
print(centered_cov)