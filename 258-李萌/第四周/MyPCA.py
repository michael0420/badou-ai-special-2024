# -*- coding: utf-8 -*-

'''使用PCA方法求样本矩阵X的K阶降维矩阵U'''

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

class All_Data(object):
    def __init__(self, X):
        # 原矩阵
        self.X = X
        # 降阶维度K
        self.K = K
        # 中心化矩阵
        self.C = None
        # 中心化协方差矩阵
        self.D = None
        # 降维转换矩阵U
        self.U = None
        # 目标降维矩阵Z
        self.Z = None

    # 矩阵X中心化
    def _Centralization(self):
        # 样本集的特征均值
        mean = np.array([np.mean(i) for i in self.X.T])
        print("原矩阵维度均值\n", mean)
        # mean1 =self.X.mean(axis=0)
        # print("原矩阵维度均值\n", mean1)
        self.C = self.X - mean
        print("中心化矩阵\n", self.C)
        return self.C
    # 协方差矩阵
    def _Covariance(self):
        m=np.shape(self.X)[0]
        print("样本数\n", m)
        self.D=np.dot(self.C.T,self.C)/m
        print("协方差矩阵\n", self.D)
        return self.D
    # 降维特征矩阵
    def _Featurearray(self):
        # 协方差矩阵的特征值和特征向量
        F_value, F_vector = np.linalg.eig(self.D)
        print("特征值\n", F_value)
        print("特征向量\n", F_vector)
        # 给出特征值降序的topK的索引序列
        value = np.argsort(-1 * F_value)
        print("降序特征值\n", value)
        # 构建K阶降维特征矩阵
        # UT = [F_vector[:, value[i]] for i in range(self.K)]
        # print('转换矩阵UT:\n', UT)
        # self.U = np.transpose(UT)
        self.U =F_vector[:, value[:self.K]]
        print('%d阶降维转换矩阵U:\n' % self.K, self.U)
        return self.U
    # 目标降维矩阵
    def _Dimreducearray(self):
        self.Z = np.dot(self.C, self.U)
        print('目标降维矩阵Z:\n', self.Z)
        return self.Z

    # PCA将原矩阵转换为K阶降维矩阵
    def _MyPCA(self):
        self.C = self.X - self.X.mean(axis=0)
        self.D = np.dot(self.C.T, self.C) / (self.C.shape[0])
        F_value, F_vector = np.linalg.eig(self.D)
        value = np.argsort(-1 * F_value)
        self.U = F_vector[:, value[:self.K]]
        self.Z = np.dot(self.C, self.U)
        print('目标降维矩阵Z:\n', self.Z)
        return self.Z

    # 使用 sklearn 接口求 PCA 降维矩阵
    def _PCAInterfaces(self):
        self.I_pca = PCA(n_components=self.K)
        self.I_pca.fit(self.X)
        self.newX = self.I_pca.fit_transform(self.X)
        print('目标降维矩阵newX:\n', self.newX)
        return self.newX

    # 清零X,Y轴散点
    def _ScatterClear(self):
        self.r_ScatterX, self.r_ScatterY, = [], []
        self.b_ScatterX, self.b_ScatterY, = [], []
        self.g_ScatterX, self.g_ScatterY, = [], []

        self.r_ScatterX.clear()
        self.r_ScatterY.clear()

        self.b_ScatterX.clear()
        self.b_ScatterY.clear()

        self.g_ScatterX.clear()
        self.g_ScatterY.clear()

    # 绘制散点图
    def _Scatterdraw(self):
        for i in range(len(self.newX)):
            if self.Y[i]==0:
                self.r_ScatterX.append(self.newX[i, 0])
                self.r_ScatterY.append(self.newX[i, 1])
                # self.r_ScatterX=self.newX[i, 0]
                # self.r_ScatterY=self.newX[i, 1]
                # plt.scatter(self.r_ScatterX, self.r_ScatterY, c='r', marker='x')
            elif self.Y[i] == 1:
                self.b_ScatterX.append(self.newX[i, 0])
                self.b_ScatterY.append(self.newX[i, 1])
                # self.b_ScatterX = self.newX[i, 0]
                # self.b_ScatterY = self.newX[i, 1]
                # plt.scatter(self.b_ScatterX, self.b_ScatterY, c='b', marker='D')
            elif self.Y[i] == 2:
                self.g_ScatterX.append(self.newX[i, 0])
                self.g_ScatterY.append(self.newX[i, 1])
                # self.g_ScatterX = self.newX[i, 0]
                # self.g_ScatterY = self.newX[i, 1]
                # plt.scatter(self.g_ScatterX, self.g_ScatterY, c='g', marker='.')

        plt.scatter(self.r_ScatterX, self.r_ScatterY, c='r', marker='x')
        plt.scatter(self.b_ScatterX, self.b_ScatterY, c='b', marker='D')
        plt.scatter(self.g_ScatterX, self.g_ScatterY, c='g', marker='.')
        plt.show()

    # --------------导出接口--------------
    # 实现PCA功能
    def _Mypca(self, K):
        self.C = self._Centralization()
        self.D = self._Covariance()
        self.U = self._Featurearray()
        self.Z = self._Dimreducearray()

    # 实现PCA功能
    def _PCA(self,K):
        self._MyPCA()

    # 鸢尾花实例
    def _Irisflower(self,K):
        self.X, self.Y = load_iris(return_X_y=True)
        self.newX=self._PCAInterfaces()
        self._ScatterClear()
        self._Scatterdraw()


'''随机矩阵应用PCA'''
if __name__ == "__main__":

    # 产生随机矩阵
    low_data = 0
    hige_data = 10
    size_data = (10, 5)
    X = np.random.randint(low_data, hige_data, size_data)
    print("随机数组X\n", X)
    K=2

    pca = All_Data(X)
    # 实现本地PCA
    pca._Mypca(K)
    pca._PCA(K)

    # 调用sklearn函数实现PCA
    pca._PCAInterfaces()

    # 鸢尾花测试
    pca._Irisflower(K)



