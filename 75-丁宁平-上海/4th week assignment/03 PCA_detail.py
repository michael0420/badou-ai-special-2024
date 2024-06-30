'''
使用PCA求样本矩阵X的K阶降维矩阵Z
'''

import numpy as np

class CPCA(object):
    '''
    用PCA求样本矩阵X的K阶降维矩阵
    Note：请保证输入的样本矩阵X shape=(m,n)  m行样例,n个特征
    '''
    def __init__(self,X,K):
        '''
        :param X: 样本矩阵X
        :param K: X的降维矩阵的阶数,即X要特征降维成K阶
        '''
        self.X = X            # 样本矩阵X
        self.K = K            # K阶降维矩阵的K值
        self.centrX = []      # 矩阵X的中心化
        self.C = []           # 样本集的协方差矩阵C
        self.U = []           # 样本矩阵X的降维转换矩阵U
        self.Z = []           # 样本矩阵X的降维矩阵Z

        self.centrX=self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()         #Z=XU求得

    def _centralized(self):
        '''矩阵的中心化'''
        print('样本矩阵X:\n',self.X)                                # 打印原样本矩阵
        centrX = []                                                # 创建一个空列表
        mean1 = np.array([np.mean(attr) for attr in self.X.T])     # 样本集的特征均值 self.X.T矩阵转置 按行遍历均值然后将均值输出为数组
        mean = np.mean(self.X,0)                                   # 0对每列求平均值,1对每行求平均值
        print('样本集的特征均值:\n',mean)                            # 打印每列特征均值
        centrX = self.X - mean                                     # 样本集的中心化：原始数据减去均值
        print('样本矩阵X的中心化centrX:\n',centrX)                   # 打印中心化以后的矩阵
        return centrX                                              # 返回中心化以后的矩阵

    def _cov(self):
        ''' 求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]      # np.shape()函数：一维时：返回元素个数，二维时，返回数组的行数和列数;此处返回的行数10(样本总数)
        # 样本矩阵的协方差矩阵C
        # n-1的使用增加了方差的估计值，以补偿只用样本数据而无法完全捕捉到总体方差的问题。
        # 样本均值直接除以n是因为我们的目标是计算平均值，而使用n-1是为了在计算样本方差时得到一个更准确的总体方差估计
        # 需要指出的是，从方差看，总体方差的分母为n，而样本方差的分母却为n - 1（自由度），这是因为当我们用n - 1为自由度的样本方差去估计总体方差时，它恰好是的无偏估计量。
        # 为什么样本标准差使用被称为自由度的n - 1，而总体的标准差使用n呢？这是因为自由度是指一组数据中可以自由取值的个数，当样本数据的个数为n时，其样本均值是确定的，只有n - 1
        # 个数据可以自由取值，其中必有一个数据不能自由取值。所以，样本的标准差只能除以n - 1，而不能除以n。如：假定一个样本有3个数值4、5、9，它的样本均值 = 6，当我们自由取值4和9时，另一个数据就不能自由取值了，它必然取5这个数字。
        # 在一个统计样本中，其标准差越大，说明它的各个观测值分布的越分散，它的集中趋势就越差。反之，其标准差越小，说明它的各个观测值分布得越集中，它的集中趋势就越好。
        C = np.dot(self.centrX.T,self.centrX)/(ns-1)               # 总体方差和样本方差的区别:自由度由n变成n-1，这里是样本方差
        print('样本矩阵X的协方差矩阵C:\n', C)                        # 打印协方差矩阵C
        return C                                                  # 返回协方差矩阵C

    def _U(self):
        '''求X的降维转换矩阵U，shape=(n,k),n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        a,b=np.linalg.eig(self.C)                  # 特征值赋值给a 对应特征向量赋值给b
        print('样本集的协方差矩阵C的特征值:\n',a)     # 打印协方差矩阵C的特征值
        print('样本集的协方差矩阵C的特征向量:\n',b)   # 打印协方差矩阵C的特征向量
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1*a)  #-1*a or -a均可    # np.argsort()函数:返回的是元素值从小到大排序后的索引值的数组
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)    #对UT进行转置
        print(f'{self.K}阶降维转换矩阵U',U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z，shape=(m,k),n是样本总数,k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X,self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__=='__main__':
    '10样本3特征的样本集，行为样本，列为特征维度'
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
    K = np.shape(X)[1]-1     # 列数(维度)-1
    print('样本集(10行3列,10个样例,每个样例3个特征):\n', X)
    pca=CPCA(X,K)