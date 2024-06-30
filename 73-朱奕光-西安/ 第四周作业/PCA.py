import numpy as np


class PCA(object):
    def __init__(self, X, K):
        self.X = X  # 原始数组X
        self.C = self.Cov()  # 协方差矩阵
        self.K = K  # 降维矩阵维度

    def Center(self):
        mean = np.mean(self.X, axis=0)  # 计算矩阵均值
        self.X = self.X - mean
        print('中心化后矩阵\n', self.X)
        return self.X

    def Cov(self):
        ns = np.shape(self.X)[0]
        cov_r = np.dot(self.X.T, self.X) / (ns - 1)
        print('协方差矩阵\n', cov_r)
        return cov_r

    def Eig(self):
        """
        计算降维转换矩阵
        """
        i, j = np.linalg.eig(self.Cov())
        a = np.argsort(-i)
        print('索引排序后结果\n', a)
        UT = [j[:, a[i]] for i in range(self.K)]  # 计算转换矩阵
        U = np.transpose(UT)
        print("转换矩阵为\n", U)
        return U

    def Result(self):
        X_res = np.dot(self.X, U)
        print('PCA算法后矩阵为\n', X_res)
        return X_res


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
K = np.shape(X)[1] - 1
pca = PCA(X, K)
X = pca.Center()
C = pca.Cov()
U = pca.Eig()
result = pca.Result()