
import numpy as np
# 1.求矩阵的中心化矩阵(计算各个列的均值，再减去对应列均值，使其均值为0得到中心化矩阵)
# 2.求矩阵中心化的协方差矩阵
# 3.通过协方差矩阵求特征值
# 4.通过特征值求特征向量
# 5.根据特征值和特征向量 得到降维后的矩阵
class CPCA(object):
    # X样本矩阵 K降维的维度
    def __init__(self,X,K):
        self.X = X #样本矩阵
        self.K = K #降维后的维度
        self.center_X = [] #样本矩阵中心化
        self.C = [] #协方差矩阵
        self.U = [] #特征向量矩阵
        self.Z = [] #降为后的矩阵

        self.center()
        self._cov()
        self._U()
        self._Z()

    # 矩阵中心化
    def center(self):
        # 求矩阵列的均值 self.X.T:样本矩阵X的转置  将列转置为行  遍历每一行的值再求出均值
        avgX = np.array([np.mean(attr) for attr in self.X.T])
        print("样本矩阵X的特征均值:\n",avgX)
        # 样本矩阵减列均值得到中心化矩阵
        self.center_X = self.X - avgX
        print("样本矩阵X的中心化矩阵:\n", self.center_X)


    # 求协方差矩阵
    def _cov(self):
        #协方差矩阵等于中心化矩阵*中心化矩阵转置*样本数分之一
        m = np.shape(self.center_X)[0] #样本数m
        self.C = np.dot(self.center_X.T,self.center_X)/(m - 1)
        print('样本矩阵X的协方差矩阵C:\n', self.C)

    # 求特征值矩阵
    def _U(self):
        # 调用求特征值函数得到特征值a和特征向量b
        a,b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 根据a的特征值大小倒序排序索引序列
        ind = np.argsort(-1*a)
        # 构建降维的特征向量转换矩阵
        UT = [ b[:,ind[i]] for i in range(self.K)]
        self.U = np.transpose(UT)
        print('特征值矩阵U:\n', self.U)
    # 求降维矩阵
    def _Z(self):
        # 根据降维维数和特征向量矩阵 * 样本矩阵 得到降维后的矩阵
        self.Z = np.dot(self.X,self.U)
        print('样本矩阵降维矩阵Z:\n', self.Z)

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)

