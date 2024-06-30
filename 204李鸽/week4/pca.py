import numpy as np

class CPCA:
    '''
    note:请确保输入的样本矩阵X.shape=(m, n)中有m行样本，n列特征
    '''
    def __init__(self, X, K ):
        '''
        功能：使用PCA求样本矩阵X的K阶降维矩阵Z
        :param X: 输入的样本矩阵X
        :param K: 矩阵X要降维成K阶
        '''
        self.X = X     # 样本矩阵X
        self.K = K     # K阶降维矩阵的K值
        self.centrX = []     # 对原始数据做零均值化（中心化）
        self.C = []     # 中心化后的协方差矩阵
        self.W = []     # K个特征列向量矩阵（特征转换矩阵）
        self.Z = []     # X的最终降维矩阵

        self.centrX = self.__centralized()
        self.C = self.__covariance()
        self.W = self.__W()
        self.Z = self.__Z()
    def __centralized(self):
        '''矩阵X的中心化'''
        print(f"样本矩阵：\n", self.X)
        centrX = []
        mean = np.mean(self.X, axis=0)
        print("样本集的特征均值：\n", mean)
        centrX = self.X - mean
        print("中心化后的样本矩阵：\n", centrX)
        return centrX
    def __covariance(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集的样例个数num
        num = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C代入协方差公式
        C = np.dot(np.transpose(self.centrX), self.centrX) / (num - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C
    def __W(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        #先求X的协方差矩阵C的特征值a和特征向量b:linalg函数是一个方言，可直接求出a,b
        a, b = np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列（排序后的索引号数组）
        ind = np.argsort(-1 * a)
        # 构建K阶降维转换矩阵M
        M =[ b[:, ind[i]] for i in range(self.K)]
        # 这个M矩阵是长度为2的列表，列表里面有shape为两个列向量，例如arr =[[2, 5, 6] ,[3, 1, 4]]
        # 写成np数组是两行三列，我们想要的W是三行两列
        W = np.transpose(M)
        print(f'{self.K}阶降维转换矩阵U:\n',  W)
        return W
    def __Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), m是样本总数，k是降维矩阵中特征维度总数'''
        Z = self.X @ self.W
        print('X_shape', np.shape(self.X))
        print('W_shape', np.shape(self.W))
        print('Z_shape', np.shape(self.Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
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
    pca = CPCA(X, K)
















