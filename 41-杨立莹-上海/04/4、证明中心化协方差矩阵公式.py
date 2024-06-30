# 证明中心化协方差矩阵公式
import numpy as np

# 数据
X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
])

# 计算样本数量
m = X.shape[0]
# 计算样本均值
mean = np.mean(X, axis=0)
# 去中心化
X_centered = X - mean
# 计算协方差矩阵（使用公式）
covariance_matrix_formula = (1/m) * np.dot(X_centered.T, X_centered)
print("使用公式计算的协方差矩阵：")
print(covariance_matrix_formula)

# 计算协方差矩阵（使用 np.cov 函数）
covariance_matrix_np = np.cov(X_centered, rowvar=False, bias=True)
print("\n使用 np.cov 函数计算的协方差矩阵：")
print(covariance_matrix_np)