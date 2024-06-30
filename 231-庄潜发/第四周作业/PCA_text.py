"""
@Author: zhuang_qf
@encoding: utf-8
@time: 2024/4/20 10:53
"""
import numpy as np


class CPCA(object):

    def __init__(self, X, K):
        self.X = X  # 样本矩阵
        self.K = K  # 我们要降的维度
        self.centerX = []  # 中心化矩阵
        self.C = []  # 协方差矩阵
        self.U = []  # 降维转换矩阵U
        self.Z = []  # 降维后的矩阵U

    def center(self):
        # 求矩阵均值, 因为要求的是列的均值, 所以原矩阵需要进行转置
        mean = np.array([np.mean(i) for i in np.transpose(self.X)])
        print("样本矩阵X的均值mean:\n", mean)
        # 矩阵中心化后的矩阵
        self.centerX = self.X - mean
        print("矩阵中心化后的矩阵:\n", self.centerX)
        # 求协方差矩阵 D = 1/(m) Z.T * Z  m样本总数
        m = np.shape(self.X)[0]
        # self.C = (np.transpose(self.centerX)*self.centerX)/(m)
        self.C = np.dot(np.transpose(self.centerX), self.centerX)/m
        print("协方差矩阵:\n", self.C)
        # 求协方差的特征值和特征向量, a为特征值, b为特征向量
        a, b = np.linalg.eig(self.C)
        print("协方差矩阵特征值:\n", a)
        print("协方差矩阵特征向量:\n", b)
        # 对特征值进行降序
        ind = np.argsort(-a)
        print("降序后的特征值索引:\n", ind)
        # 构建降维转换矩阵U
        self.U = b[:,ind[:self.K]]
        print("降维转换矩阵U:\n", self.U)
        # 计算降维后的矩阵Z=XU  X为中心化后的矩阵
        self.Z = np.dot(self.centerX, self.U)
        print("降维矩阵Z:\n", self.Z)


if __name__ == '__main__':
    X = np.array([[8, 4],
                  [1, 2],
                  [6, 12]])
    K = np.shape(X)[1] - 1
    print("样本矩阵:\n", X)
    pca = CPCA(X, K)
    pca.center()
