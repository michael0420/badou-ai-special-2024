from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# 创建数据
A_exact = 20 * np.random.random((500, 1))
perfect_fit = 60 * np.random.normal(size=(1, 1))
B_exact = np.dot(A_exact, perfect_fit)

# 加入高斯噪声
A_noisy = A_exact + np.random.normal(size=A_exact.shape)
B_noisy = B_exact + np.random.normal(size=B_exact.shape)

# 使用 RANSAC 算法拟合模型
ransac = RANSACRegressor(base_estimator=LinearRegression(), min_samples=50, max_trials=100, loss='absolute_loss')
ransac.fit(A_noisy, B_noisy)

# 预测
predicted = ransac.predict(A_noisy)

# 绘图
plt.scatter(A_noisy, B_noisy, color='yellowgreen', marker='.', label='Inliers')
plt.plot(A_noisy, predicted, color='navy', linewidth=2, label='RANSAC regressor')
plt.legend()
plt.xlabel("Input")
plt.ylabel("Response")
plt.title("RANSAC Regression")
plt.show()
