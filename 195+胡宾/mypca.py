import numpy as np


class PCA_KK(object):
    def __init__(self, array, K):
        # array 原矩阵
        # 目标维度
        self.X = array
        self.K = K
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self.zhongxinhua()
        self.C = self.covdxd()
        self.U = self._numda()
        self.Z = self._Result()  # Z=XU求得

    def zhongxinhua(self):
        "样本矩阵中心化"
        print('样本矩阵X:\n', self.X)
        zhongxinghuajuzheng = []
        zhongxingzhi = np.array([np.mean(i) for i in self.X.T])
        print('样本集的特征均值:\n', zhongxingzhi)
        zhongxinghuajuzheng = self.X - zhongxingzhi
        print('样本矩阵X的中心化centrX:\n', zhongxinghuajuzheng)
        return zhongxinghuajuzheng

    def covdxd(self):
        "求样本矩阵的协方差矩阵"
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵Z
        Z = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', Z)
        return Z

    def _numda(self):
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)
        # 给出特征值降序的topK的索引序列
        index = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵_U
        _U = [b[:, index[i]] for i in range(self.K)]
        U = np.transpose(_U)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Result(self):
        C = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(C))
        print('样本矩阵X的降维矩阵Z:\n', C)
        return C


if __name__ == '__main__':
    array = np.array([[100, 15, 29],
                  [25, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [18, 25, 35],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(array)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', array)
    # pca算法用于降维
    my_pca = PCA_KK(array, K)
