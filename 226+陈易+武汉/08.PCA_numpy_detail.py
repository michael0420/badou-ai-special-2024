# -*- coding:utf-8 -*-
# import numpy as np
"""
实用PCA求样本矩阵X的K介降维矩阵Z
"""
import numpy as np
class CPCA(object):
    """
    用PCA求样本矩阵X的K介降维矩阵Z
    Note：请保证输入的样本矩阵X  shape=(m,n)，m 行样例，n个特征
    """
    def __init__(self, X, K):
        """
        :param X: 样本矩阵X
        :param K: X的降维矩阵的阶数，即X要特征降维成K阶
        """
        self.X = X          # 样本矩阵
        self.K = K          # K阶降维矩阵的K值
        self.centrX = []    # 矩阵X的中心化
        self.C = []         # 样本集的协方差矩阵C
        self.U = []         # 样本矩阵X的降维转置矩阵
        self.Z = []         # 样本矩阵X的降维矩阵Z
        # 开始计算
        self.centrX = self._centralized()       # 中心化，调用中心化函数
        self.C = self._cov()                    # 求协方差矩阵
        self.U = self._U()                      # 求样本矩阵X的降维转置矩阵
        self.Z = self._Z()                      # Z = X * U
    def _centralized(self):
        """矩阵X的中心化"""
        print("样本矩阵X：\n",self.X)
        centrX = []
        # 计算特征均值。  np.array：可以从python列表创建数组   np.mean：用于计算数组中元素的平均值
        mean = np.array([np.mean(attr) for attr in self.X.T])    # 样本集的特征均值
        print("样本集的特征均值：\n",mean)
        centrX = self.X - mean           # 样本集的中心化。样本数值 - 特征均值
        print("样本矩阵X的中心化centrX：\n",centrX)
        return centrX

    def _cov(self):
        """求样本矩阵X的协方差矩阵C"""
        # 样本集的样例总数      shape函数的功能是读取矩阵的长度，比如shape[0]就是读取矩阵第0维度的长度,相当于行数，是图片的高度
                                                          #shape[1]就是读取矩阵第1维度的长度,相当于列数，是图片的宽度
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  # dot求两个向量的积， ns -1为无偏估计
        print("样本矩阵X的协方差矩阵为C:",C)
        return C

    def _U(self):
        """求X的降维转置矩阵U，shape(n,k) n是X的特征维度总数  k是降维矩阵的特征维度"""
        a,b = np.linalg.eig(self.C)  # 特征值赋值给a，对应特征向量赋值给b。
        print("样本集的协方差矩阵C的特征值：\n",a)
        print("样本集的协方差矩阵C的特征向量：\n",b)
        # 给出特征值降序的topK 的索引序列
        ind = np.argsort(-1 * a)
        # 构件K阶降维的降维转置矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print("%d阶降维转换矩阵U：\n" %self.K,U)        # 为什么要转置
        return U

    def _Z(self):
        """按照Z=XU求降维矩阵Z"""
        Z = np.dot(self.X, self.U)
        print("样本矩阵X的降维矩阵Z：\n",Z)
        return Z

if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)