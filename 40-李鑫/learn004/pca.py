import numpy as np

X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
# 计算特征均值
mean = np.array([np.mean(attr) for attr in X.T])
# 样本集中心化
X_centre = X - mean
# 协方差矩阵
C = np.dot(X_centre.T, X_centre)/(np.shape(X)[0] - 1)
# 求协方差矩阵的特征值和特征向量
a, b = np.linalg.eig(C)
# 特征值降序排列
ind = np.argsort(-1*a)
# 取降维数的特征向量并转换为矩阵
UT = [b[:, ind[i]] for i in range(2)]
U = np.transpose(UT)
# 求出降维矩阵
Z = np.dot(X, U)
print(X, X_centre, U, Z, sep="\n*************************\n")
