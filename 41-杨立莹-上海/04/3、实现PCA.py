# 实现PCA
import numpy as np

def pca(X, k):
    # 计算样本均值
    mean = np.mean(X, axis=0)
    # 去中心化
    X_centered = X - mean
    # 计算协方差矩阵
    cov_matrix = np.dot(X.T,X)/X.shape[0]
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # 选取前 k 个特征向量
    top_k_eigenvectors = eigenvectors[:, -k:]
    # 投影到新的特征空间
    transformed_data = np.dot(X_centered, top_k_eigenvectors)
    # 返回降维后的数据
    return transformed_data

# 数据
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# 设置要保留的主成分数量
k = 2
# 调用PCA函数进行降维
transformed_data = pca(data, k)
print("原始数据：")
print(data)
print("\n降维后的数据：")
print(transformed_data)