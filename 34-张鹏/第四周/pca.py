import numpy as np  
  
def pca(X, k):  
    # 1. 计算均值并中心化数据  
    mean = np.mean(X, axis=0)  
    X_centered = X - mean  
  
    # 2. 计算协方差矩阵  
    cov_matrix = np.cov(X_centered.T)  #'.T' 获取数组的转置
  
    # 3. 计算协方差矩阵的特征值和特征向量  
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  
  
    # 4. 对特征值进行排序，并选取前k个特征值对应的特征向量  
    idx = eigenvalues.argsort()[::-1]  
    eigenvectors = eigenvectors[:, idx]  
    k_eigenvectors = eigenvectors[:, :k]  
  
    # 5. 将原始数据转换到新的k维空间中  
    X_pca = X_centered.dot(k_eigenvectors)  
  
    return X_pca  
  
# 示例  
# 一个形状为(n_samples, n_features)的二维数组X，你想要将其降低到k维  
X = np.random.rand(100, 3)  # 100个样本，每个样本3个特征  
k = 2  # 我们想要降低到2维  
X_pca = pca(X, k)  
print(X_pca.shape)  # 输出应该是 (100, 2)
# print(X_pca)