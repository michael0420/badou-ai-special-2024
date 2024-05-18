import numpy as np
X = np.array([    [10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])

T = X - X.mean(axis=0)
C = np.cov(X.T)
print('矩阵的中心化:\n', T)
print('协方差矩阵C:\n',C)

w,v = np.linalg.eig(C)
print('协方差矩阵C的特征值：\n',w)
print('协方差矩阵C的特征向量:\n',v)
# 获得降序排列特征值的序号
idx = np.argsort(-w)
# 降维矩阵,n维度
n=2
Y = v[:,idx[:n]]
print('降维后的矩阵：\n',np.dot(X,Y))
