import numpy as np

class PCA(object):
    def __init__(self,X,K):
        #样本矩阵X
        #矩阵X的降维矩阵的阶数
        self.X = X
        self.K = K
        self.centrX = []
        self.xiefangchaC = []
        self.zhuanhuanU = []
        self.jiangweiZ = []

        self.centrX = self._centralized()
        self.xiefangchaC = self.xiefangchaC()
        self.zhuanhuanU = self.zhuanhuanU()
        self.jiangweiZ = self.jiangweiZ()

    #矩阵X中心化
    def _centralized(self):
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T]) #求均值
        centrX = self.X - mean
        return centrX

    #求协方差矩阵
    def _xiefangchaC(self):
        ns = np.shape(self.centrX)[0]
        xiefangchaC = np.dot(self.centrX.T,self.centrX)/(ns - 1)
        return xiefangchaC

    def _zhuanhuanU(self):
        a,b = np.linalg.eig(self.xiefangchaC)
        #排序
        ind = np.argsort(-1*a)
        zhuanhuanUT = [b[:,ind[i]] for i in range(self.K)]
        zhuanhuanU = np.transpose(zhuanhuanUT)
        return zhuanhuanU

    def _jiangweiZ(self):
        jiangweiZ = np.dot(self.X,self.zhuanhuanU)
        return jiangweiZ

if __name__ == '__main__':
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


