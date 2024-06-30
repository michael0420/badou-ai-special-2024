'''
PCA手算降维，细节版
'''

import numpy as np


class PCA(object):
    def __init__(self, X, K):  # 初始化参数，搭积木工具准备
        '''
        初始化方法，定义各种需要的变量，需要保证输入的样本矩阵X.shape=(m, n)，m行样例，n个特征
        :param X: 源数据
        :param K: 准备降维的阶数
        首先准备各种砖头
        '''
        self.X = X  # 样本矩阵 mxn
        self.K = K  # 降维至K维
        self.m, self.n = X.shape[0], X.shape[1]  # 样本为m行，特征维度为n列
        # 过程变量矩阵初始化，可以省略，这里方便程序逻辑
        self.center_X = []  # 1.样本矩阵中心化处理 mxn
        self.cov_center_X = []  # 2.已中心化样本矩阵的协方差矩阵 nxn
        self.base_vector_X = []  # 3.根据协方差矩阵，找前K个特征值对应特征向量组成特征空间，即降K维投影的基向量矩阵,以进行矩阵映射变换 nxk
        self.new_X = []  # 4.降维后矩阵，结束

        # 依次进行中心化、协方差、特征向量、映射函数计算
        self.center_X = self.centralize()  # 中心化，需要定义好centralize函数(后面可以调用，不然函数内部的center_X不能调用)
        self.cov_center_X = self.cal_covariance()
        self.base_vector_X = self.base_vector()
        self.new_X = self.project()

    def centralize(self):  # 1.样本矩阵中心化处理
        print('10样本3特征的样本集为：\n', self.X)
        mu = np.array([np.mean(attr) for attr in self.X.T])  # attr按self.X.T的行取出，求同一特征维度应转置 1xn
        # mu = np.array([np.mean(self.X, axis=0)])  # axis=0直接按列取均值更好理解
        print('各特征维度均值形成向量为：\n', mu)
        center_X = np.array([self.X[i] - mu for i in range(self.m)])  # 所有特征减相应均值，使mu=0, mxn
        # center_X = self.X - mu  # 更简单，矩阵与矩阵（同维度）或向量（一维）运算，其他维度不等会value error
        print('经中心化处理的样本矩阵为：\n', center_X)
        return center_X

    def cal_covariance(self):  # 2.已中心化样本矩阵的协方差矩阵，C=1/(n-1)*X.T*X
        cov_center_X = 1 / (self.m - 1) * np.dot(self.center_X.T, self.center_X)  # 样本数m-1是无偏估计 nxn
        print('协方差矩阵为：\n', cov_center_X)
        return cov_center_X

    def base_vector(self):  # 3.协方差矩阵的前K阶特征向量,np.argsort和np.linale.eig的用法
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_center_X)  # 自动计算特征值赋值给a，对应特征向量赋值给b
        # https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('协方差矩阵特征值为：\n', eigenvalues)
        print('协方差矩阵特征向量为：\n', eigenvectors)  # 对应特征向量按列
        index = np.argsort(-1 * eigenvalues)  # 特征值的从大到小，降序K的索引序列
        print(index)
        # temp = np.transpose(eigenvectors)
        # base_vector_XT = np.array([temp[j] for j in range(self.K)])  # 此式计算方法有误，本例index排序恰好为0， 1， 2，temp未考虑大小排序

        base_vector_XT = [eigenvectors[:, index[i]] for i in range(self.K)]  # 注意特征向量以行形式展现，应转置
        base_vector_X = np.transpose(base_vector_XT)  # nxk
        print('前%d阶特征向量组成的投影基为：\n' % self.K, base_vector_X)
        return base_vector_X

    def project(self):  # 4.降维后矩阵，结束
        new_X = np.dot(self.X, self.base_vector_X)  # mxn * nxk ->mxk，m个样本k个特征的降维
        print('样本矩阵降维至%d维后矩阵为：\n' % self.K, new_X)
        return new_X


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    A = np.array([[10, 15, 29],
                 [15, 46, 13],
                 [23, 21, 30],
                 [11, 9,  35],
                 [42, 45, 11],
                 [9,  48, 5],
                 [11, 21, 14],
                 [8,  5,  15],
                 [11, 12, 21],
                 [21, 20, 25]])
    PCA(A, 2)
