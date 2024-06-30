# coding = utf-8

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

# 输入原始数据
# X = np.random.randint(0, 255, size=(7, 6))
X = np.array([
 [204, 162, 219, 144, 203, 154],
 [ 61, 207,  40, 194, 160,   4],
 [211, 105, 167, 112, 142, 189],
 [ 48, 225, 167, 173,  86,  55],
 [129,  19, 194,  63,  87, 187],
 [ 46, 230, 161, 151, 254, 124],
 [ 71,  33, 108,  10,  69,  97]
])

# 使用sklearn接口实现PCA
# p = PCA(n_components=3)  # 设置降维后维度为3
# p.fit(X)                 # 执行降维
# X_new = p.transform(X)   # 输出降维结果
# print('sklearn实现目标降维矩阵：\n', X_new)
# print('*'*80)


class PCA(object):

    def __init__(self, X, k):
        self.X = X          # 参数原始矩阵X
        self.k = k          # 参数目标维度k
        self.center = []    # 存储中心化矩阵
        self.A = []         # 存储协方差矩阵
        self.B = []         # 存储特征向量矩阵
        self.C = []         # 目标降维矩阵

        self.center = self.center_fun()      # 求中心化矩阵
        self.A = self.cov()                 # 求协方差矩阵
        self.B = self.feature()             # 求需要的特征向量矩阵
        self.C = np.dot(self.center, self.B)     # 中心化矩阵与选取后的特征向量矩阵乘积
        print('目标降维矩阵：\n', self.C)

    def center_fun(self):
        avg = np.mean(self.X, axis=0)  # 计算每列（每个特征）的均值
        print('样本集的特征均值:\n', avg)
        center = self.X - avg  # 中心化矩阵
        print('中心化后的矩阵：\n', center)
        return center

    def cov(self):
        A = np.cov(self.center.T)  # 计算协方差矩阵
        print('协方差矩阵：\n',  A)
        return A

    def feature(self):
        a, b = np.linalg.eig(self.A)                         # 求协方差矩阵的特征值a，特征向量b
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        li = np.argsort(-a)                                  # 特征值降序排序
        BT = [b[:, li[index]] for index in range(self.k)]    # 选择前k位的特征向量矩阵
        B = np.array(BT).T  # 将列表转换为数组并转置
        print('选取的目标特征向量矩阵：\n', B)
        return B

# if __name__=='__init__':
# 在范围内随机设置降维后维度，random函数为闭区间


k = random.randint(1, np.shape(X)[1])
print("手动实现PCA降维后维度为：", k)
pca = PCA(X, k)