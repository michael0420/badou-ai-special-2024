#pca 主成分分析
'''
1 获取均值
2 中心化
3 计算协方差
4 计算协方差特征值和特征向量
5 计算原矩阵对协方差矩阵的映射
'''

import numpy as np
import cv2

class PCA(object):
    #初始化
    def __init__(self,X,K):
        self.X = X  #原矩阵
        self.K = K  #保留特征数量
        self.avg = []  #均值，原矩阵列的平均数
        self.centored=[]  #中心化，用原矩阵X-avg得到
        self.C = []  # 协方差矩阵
        self.T = [] # 求降维特征矩阵
        self.Z = [] #降维矩阵

        self.avg = self._avg()
        self.centored = self._centored()
        self.C = self._cov()
        self.T = self._T()
        self.Z = self._Z()


    #计算均值
    def _avg(self):
        #axis=0 表示列均值，axis=1 行均值
        avg = np.mean(self.X, axis=0)
        #self.avg = np.array([np.mean(d) for d in self.X.T])

        print("样本X均值=\n", avg)
        return avg

    #计算中心化
    def _centored(self):
        centored = self.X - self.avg
        print("样本x中心化矩阵=\n",centored)
        return centored

    # 协方差矩阵
    def _cov(self):
        #1/m * ZT * Z
        m = np.shape(self.centored)[0] - 1
        cov = np.dot(self.centored.T,self.centored)/m
        print("样本X的协方差矩阵=\n",cov)
        return cov

    def _T(self):
        #计算矩阵的特征值，和特征向量，a为特征值，b为特征矩向量
        a, b = np.linalg.eig(self.C)
        #使用argsort得到排序后的索引数组
        sora = np.argsort(-a)
        print("样本X的协方差矩阵C的特征值排序:\n", sora)
        DT = np.array([b[:,sora[i]] for i in range(self.K)])
        print("原始特征向量=\n",DT)

        D = np.transpose(DT)
        print("特征向量=\n", D)
        return D

    def _Z(self):
        #计算最终结果
        Z = np.dot(self.X, self.T)
        print("样本矩阵X的降维矩阵Z=\n",Z)
        return Z


if  __name__ == '__main__':
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
    #获取列数量
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X, K)
