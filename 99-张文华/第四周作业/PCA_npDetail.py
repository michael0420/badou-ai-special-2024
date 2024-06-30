'''
作业3：
实现pca主成分分析
'''

import numpy as np


# 定义一个类，实现pca算法
class CPCA():

    # 定义初始化方法
    def __init__(self, data, k):
        self.data = data        # 样本矩阵数据
        self.k = k              # 降维后，矩阵的维度数，
        self.center_data = self.centralized()   # 矩阵中心化
        self.D = self._cov()             # 协方差矩阵
        self.U = self._U()             # 样本矩阵的降维转化矩阵
        self.Z = np.dot(self.data, self.U)            # 样本矩阵降维后的矩阵

    # 进行矩阵中心化
    def centralized(self):
        print('原始矩阵数据：', self.data)
        mean = np.array([np.mean(attr) for attr in self.data.T])
        print('样本的特征均值', mean)
        center_data = self.data - mean

        print('中心化后的矩阵：', center_data)
        return center_data

    # 求样本矩阵的协方差矩阵
    def _cov(self):
        # 样本集的样列总数
        ns = np.shape(self.center_data)[0]
        # 求协方差矩阵
        D = np.dot(self.center_data.T, self.center_data)/(ns - 1)
        print('协方矩阵：', D)
        return D

    # 求样本矩阵的降维转化矩阵
    def _U(self):
        # 将协方差矩阵D的特征值，特征向量分别赋值给a， b
        a, b = np.linalg.eig(self.D)
        print(f'协方差矩阵的特征值:\n{a},\n'
              f'协方差矩阵的特征向量:\n{b}')
        # 给出特征值降序的索引序列
        ind = np.argsort(-1 * a)
        print('降序的索引序列:', ind)
        UT = [b[:,ind[i]] for i in range(self.k)]
        U = np.transpose(UT)
        print(U)
        return U


if __name__ == '__main__':
    # 定义一个数组
    data = np.array([[10, 11, 15],
                     [15, 46, 77],
                     [58, 66, 77],
                     [25, 23, 45],
                     [35, 14, 58],
                     [48, 25, 99],
                     [85, 41, 33],
                     [75, 95, 12],
                     [15, 25, 69],
                     [34, 56, 89],
                     [15, 10, 28],
                     [47, 22, 66],
                     [85, 10, 91]])
    # 设定对数组维度降一维
    k = data.shape[1] - 1
    pca = CPCA(data, k)
    print(pca.Z)