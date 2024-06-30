# PCA
# 中心化
# 求特征值\特征向量 (基于协方差矩阵)
# 选取前k个特征值,截取前k个特征向量
# 原矩阵*截取后的特征向量 得到一个新的矩阵
import numpy as np


class MY_PCA():
    def __init__(self,X,K):
        self.X = X
        self.K = K
        self.centrX=[] # 矩阵X的中心化
        self.C = []
        self.U = []
        self.Z = []
        # 零均值化
        self.centrX = self.centralized()
        # 求协方差矩阵
        self.C = self.cov()
        # 求特征矩阵,并且根据K进行降维
        self.U = self._U()
        # 降维后的特征矩阵*原矩阵X
        self.Z = self._Z()


    def centralized(self):
        print("原矩阵X:\n",self.X)

        # 求均值
        # mean = []
        # for attr in self.X.T :
        #     mean.append(np.mean(attr))
        #  矩阵所有的元素减去均值
        mean = np.array([np.mean(attr) for attr in self.X.T])
        print("均值 mean:\n",mean)
        central_X = self.X - mean
        print("中心化后矩阵:\n",central_X)

        return central_X

    def cov(self):
        # 样本数量
        ns = np.shape(self.X)[0]
        C = np.dot(self.centrX.T,self.centrX)/(ns-1)
        print("协方差矩阵:\n",C)

        return C

    # 求 特征值 => 降维转换特征矩阵
    def _U(self):
        # 求特征向量和特征值
        a,b=np.linalg.eig(self.C)
        # 返回的是排序后的索引,按照升序
        index = np.argsort(-1*a)
        # 根据k,对特征向量进行降维
        print("原特征向量\n",b)

        UT = np.array([b[:,index[i]] for i in range(self.K)])

        U = UT.T
        print("降维特征向量\n",U)

        return U

    # 进行降维
    # 降维转换特征矩阵 * 原矩阵
    def _Z(self):
        Z = np.dot(self.X,self.U)
        print("降维的结果",Z)
        return Z


if __name__ == '__main__':
    X = np.array([[1, 2, 4],
                  [1, 1, 5],
                  [1, 3, 6]])
    pca = MY_PCA(X,2)
