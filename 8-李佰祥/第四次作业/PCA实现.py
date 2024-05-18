import cv2
import numpy as np

X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])

K = X.shape[1] -1   #降为2纬
#输入矩阵中心化/零均值化
#为什么要中心化？
mean = np.array([np.mean(i) for i in X.T])
centerX = X - mean
print(centerX)

#求协方差矩阵
ns = np.shape(centerX)[0]
# 样本矩阵的协方差矩阵C
C = np.dot(centerX.T, centerX) / (ns - 1)
print(C)

#求特征值与特征向量
a,b = np.linalg.eig(C)
print(a)
print(b)

ind = np.argsort(-1 * a)
# 构建K阶降维的降维转换矩阵U
UT = [b[:, ind[i]] for i in range(K)]
U = np.transpose(UT)

Z = np.dot(X, U)
print('X shape:', np.shape(X))
print('U shape:', np.shape(U))
print('Z shape:', np.shape(Z))
print('样本矩阵X的降维矩阵Z:\n', Z)











