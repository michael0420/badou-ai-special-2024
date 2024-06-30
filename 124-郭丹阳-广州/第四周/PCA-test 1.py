import numpy as np

# 数据集
X = np.array([[-1, 2, 66, -1], 
              [-2, 6, 58, -1], 
              [-3, 8, 45, -2], 
              [1, 9, 36, 1], 
              [2, 10, 62, 1], 
              [3, 5, 83, 2]])

# 中心化数据
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# 计算协方差矩阵
cov_matrix = np.cov(X_centered, rowvar=False)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 将特征向量按特征值降序排列
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# 选择前N个主成分（特征向量）
num_components = 2  # 选择前2个主成分
principal_components = eigenvectors[:, :num_components]

# 投影数据到低维空间
projected_data = np.dot(X_centered, principal_components)

print("原始数据集 X 的形状:", X.shape)
print("降维后的数据集形状:", projected_data.shape)
print("投影后的数据集:")
print(projected_data)
