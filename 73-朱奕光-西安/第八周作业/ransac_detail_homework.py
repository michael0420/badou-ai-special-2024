import numpy as np
import matplotlib.pyplot as plt

# 生成数据
n_samples = 500
n_outliers = 50

# 生成线性数据
X = np.linspace(0, 10, n_samples)
y = 3 * X + 10 + np.random.randn(n_samples) * 2

# 添加噪声点
X[:n_outliers] = np.linspace(0, 10, n_outliers)
y[:n_outliers] = 30 * np.random.randn(n_outliers)

# RANSAC 算法
def ransac(X, y, n, k, t, d):
    best_fit = None
    best_err = float('inf')
    data = np.column_stack((X, y))
    for i in range(k):
        maybe_inliers = data[np.random.choice(data.shape[0], n, replace=False), :]
        maybe_model = np.polyfit(maybe_inliers[:, 0], maybe_inliers[:, 1], 1)
        also_inliers = data[np.abs((np.polyval(maybe_model, data[:, 0]) - data[:, 1])) < t, :]
        if len(also_inliers) > d:
            better_model = np.polyfit(also_inliers[:, 0], also_inliers[:, 1], 1)
            errors = also_inliers[:, 1] - np.polyval(better_model, also_inliers[:, 0])
            this_err = np.mean(errors ** 2)
            if this_err < best_err:
                best_fit = better_model
                best_err = this_err
    return best_fit

# 最小二乘法
def least_squares(X, y):
    A = np.vstack([X, np.ones(len(X))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

# RANSAC 参数
n = 2  # 最小样本数
k = 1000  # 最大迭代次数
t = 2  # 距离阈值
d = 250  # 内点数阈值

ransac_model = ransac(X, y, n, k, t, d)
ls_model = least_squares(X, y)

# 可视化
plt.scatter(X, y, color='yellowgreen', marker='.', label='Data')
plt.plot(X, np.polyval(ransac_model, X), color='black', linewidth=2, label='RANSAC fit')
plt.plot(X, ls_model[0]*X + ls_model[1], color='blue', linewidth=2, label='Least squares fit')
plt.legend()
plt.show()
