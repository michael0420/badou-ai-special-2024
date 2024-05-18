import numpy as np
from sklearn.decomposition import PCA

# 中心化矩阵函数
def centralized(X):
    # 求矩阵平均值
    mean = np.mean(X, axis=0)
    print('X矩阵的特征均值：\n', mean)
    # 求中心化矩阵
    CE_X = X -mean
    print('中心化X矩阵：\n',CE_X)

    return CE_X

# 协方差矩阵函数,公式为D=ZT*Z/(m-1)
def cov(cen_X):
    # 获取样本数
    m = cen_X.shape[0]
    print(m)
    # 求协方差矩阵
    CO_X = np.dot(cen_X.T, cen_X)/(m-1)
    print('中心化X矩阵的协方差矩阵：\n', CO_X)
    return CO_X

# 求降维后的特征矩阵函数
def Z(X,cov_X,K):
    # 使用linalg函数获取矩阵的特征值和特征向量，函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
    a, b = np.linalg.eig(cov_X)
    print('协方差矩阵的特征值:\n', a)
    print('协方差矩阵的特征向量:\n', b)
    # 降序排序特征索引
    ind = np.argsort(a)[::-1]
    print("降序特征索引：\n", ind)
    # 按降序索引求降维特征矩阵
    U = b[:, ind[:K]]
    print('排序并且降维的特征矩阵：\n',U)
    # 求降维后的特征矩阵
    Z = np.dot(X, U)
    return Z



# 设定一个特征值为3样本为10的矩阵
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
# 设定需降维的数量
K = X.shape[1] - 1
# 求中心化矩阵
cen_X = centralized(X)
# 求协方差矩阵
cov_X = cov(cen_X)
# 求降维后矩阵
Z_X = Z(X, cov_X, K)
print("已降维数据集:\n", Z_X)

# sklearn库PCA用法
# pca = PCA(n_components=2)   #降到2维
# pca.fit(X)                  #执行
# newX=pca.fit_transform(X)