"""alt+回车自动导包"""
import numpy as np
# from numpy.typing.tests.data.fail import ndarray
from numpy import ndarray


class PCA(object):  # 继承新式类，兼容python2
    def __init__(self, X: ndarray, k: int):  # 构造方法，构建类对象就会自动执行//形参注解
        self.X = X  # 样本矩阵
        self.k = k  # k阶降维矩阵的k值
        self.centreX = []  # 中心化样本矩阵
        self.C = []  # 协方差矩阵
        self.W = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z=X*W

        self.centreX = self._centralized()
        self.C = self._cov()
        self.W = self._W()
        self.Z = self._Z()

    def _centralized(self):
        """矩阵X的中心化"""
        print("样本矩阵X：\n", self.X)
        print("样本矩阵的转置X.T:\n", self.X.T)
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centreX = self.X - mean  # 样本集的中心化
        print('样本矩阵X的中心化centreX:\n', centreX)
        return centreX

    def _cov(self):
        """求样本矩阵X的协方差矩阵C"""
        ns = np.shape(self.centreX)[0]  # 样本集的样本总数
        print("样本个数：", ns)
        C = np.dot(self.centreX.T, self.centreX)/(ns-1)  # 中心化矩阵的协方差矩阵公式，ns-1是样本的无偏估计表示
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _W(self):
        """求特征向量矩阵"""
        # 先求centerX的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)  # np.linalg.eig是NumPy中用于计算矩阵的特征值和特征向量的函数
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量矩阵:\n', b)
        # 将特征值从小到大排序，取前k个特征值对应的特征向量
        '''原始数组 arr 是 [40, 10, 30, 20]。对这个数组使用 np.argsort 函数后，
        会得到一个新的数组 [1, 3, 2, 0],这个数组表示原始数组排序后的索引值顺序。
        也就是说，按照原数组的顺序，元素10在第1个位置，元素20在第3个位置，元素30在第2个位置，元素40在第0个位置，
        索引数组 [1, 3, 2, 0] 对应的元素值就按照由小到大的顺序'''
        ind = np.argsort(-1*a)
        # 构建K阶降维的降维转换矩阵W
        WT = [b[:, ind[i]] for i in range(self.k)]
        # W = b[:, ind[:self.k]]
        print('WT:\n', WT)
        W = np.transpose(WT)
        print('%d阶降维转换矩阵W:\n' % self.k, W)
        return W

    def _Z(self):
        """按照Z=XW求降维矩阵Z, shape=(n,k), n是样本总数，k是降维矩阵中特征维度总数"""
        Z = np.dot(self.centreX, self.W)
        print('X shape:', np.shape(self.X))
        print('W shape:', np.shape(self.W))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    X = np.array([[10, 15, 29],  # 10行3列
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
    # print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X, K)










