# -*- coding: utf-8 -*-
"""
使用PCA求样本矩阵X的K阶降维矩阵Z
"""

import numpy as np


class CPCA(object):
    #定义了一个名为 CPCA的类，它包含了一些方法来实现主成分分析（PCA）并进行特征降维
    '''用PCA求样本矩阵X的K阶降维矩阵Z
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, X, K):
        '''
        :param X,样本矩阵X
        :param K,X的降维矩阵的阶数，即X要特征降维成k阶
        '''
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrX = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        '''矩阵X的中心化'''
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  ##样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX
    """
    def _centralized(self):
    '''矩阵X的中心化'''
    print('样本矩阵X:\n', self.X)  # 输出传入的样本矩阵X 
 
    centrX = []  # 初始化一个空列表来存放中心化后的样本矩阵 
 
    mean = np.array([np.mean(attr) for attr in self.X.T])  # 对每列（特征）计算平均值，得到样本集的特征均值 
    # np.mean(attr) 对attr（即self.X的一列）计算平均值 
    # [np.mean(attr) for attr in self.X.T] 列表推导式，对每列都做一次均值计算，结果为一维数组，包含了所有特征的均值 
 
    print('样本集的特征均值:\n', mean)  # 输出计算得到的特征均值 
 
    centrX = self.X - mean  # 用原始矩阵X中的每个元素减去对应的特征均值，完成矩阵的中心化 
    # self.X - mean 通过广播机制，NumPy会将mean扩展为与self.X同形，然后进行元素级相减 
 
    print('样本矩阵X的中心化centrX:\n', centrX)  # 输出中心化后的样本矩阵 
 
    return centrX  # 返回中心化后的样本矩阵 
    """
    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C
    """
    def _cov(self):
    '''求样本矩阵X的协方差矩阵C'''
    # 样本集的样例总数 
    ns = np.shape(self.centrX)[0]  # 使用np.shape获取self.centrX的形状，然后取第一个元素作为样本数 
    
    # 样本矩阵的协方差矩阵C 
    C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  # 计算协方差矩阵 
    # np.dot(a, b) 相当于 a^T * b，这里计算的是中心化样本矩阵的转置乘以自身 
    # 除以 (ns - 1) 是因为在计算协方差时我们除以的是自由度，即样本数减一 
 
    print('样本矩阵X的协方差矩阵C:\n', C)  # 输出协方差矩阵C 
 
    return C  # 返回协方差矩阵C 

    这个方法首先计算了中心化样本矩阵 self.centrX 的行数，也就是样本总数 ns。然后，它通过将 self.centrX 的转置乘以自身来计算协方差矩阵 C。
    在计算协方差矩阵时，我们需要将每个中心化样本向量的乘积加权求和，然后除以样本数减一（这是因为我们在估计总体协方差时使用的是样本协方差）
    最后，该方法输出协方差矩阵并将其返回。

    在实践中，可以使用 NumPy 的 np.cov 函数来更简洁地计算协方差矩阵，它可以接受一个矩阵并返回相应的协方差矩阵。例如：

     C = np.cov(self.centrX.T)

     这行代码将会得到与原代码相同的结果，而且更加直接易懂
     """

    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(
            self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    """
    def _U(self):
    '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
    # 先求X的协方差矩阵C的特征值和特征向量 
    a, b = np.linalg.eig(self.C)  # a包含特征值，b包含特征向量 
    # 特征值赋给变量a，对应的特征向量赋给变量b 
    print('样本集的协方差矩阵C的特征值:\n', a)
    print('样本集的协方差矩阵C的特征向量:\n', b)
 
    # 给出特征值降序排列的topK的索引序列 
    # np.argsort(a, axis=-1) 对数组a进行排序，并返回排序后的索引数组 
    # 在这里我们传入参数 -1 代表按列排序，a 被认为是列向量 
    # 通过给定的代码，我们得到了按降序排列的特征值的索引 
    ind = np.argsort(-1 * a)
 
    # 构建K阶降维的降维转换矩阵U 
    # 通过列表解析，我们从特征向量矩阵中选取了对应降序特征值的前K个特征向量 
    UT = [b[:, ind[i]] for i in range(self.K)]
    U = np.transpose(UT)  # 然后将这些特征向量组成一个矩阵，并转置得到最终的转换矩阵U 
 
    print('%d阶降维转换矩阵U:\n' % self.K, U)
    return U 

请注意，在这里我们使用了 NumPy 的 np.linalg.eig 函数来计算协方差矩阵 C 的特征值和特征向量。然后，我们通过列表解析选择了前 K 个
最大的特征值所对应的特征向量，并将它们组合成一个新的矩阵 UT。最后，我们将这个矩阵转置得到 U，即降维转换矩阵。

在实践中，为了得到更好的性能和稳定性，您可以使用 NumPy 的 np.argsort 函数的返回值来直接索引特征向量矩阵 b，
而不是通过列表解析。这样可以避免潜在的性能问题和内存使用增加。例如：

# 直接从特征向量矩阵中选取前K个特征向量 
U = b[:, np.argsort(-a)[-self.K:]]
# 转置得到最终的转换矩阵U 
U = U.T 

这样的代码会更加高效且易于阅读。
     
     """

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z
    """
    def _Z(self):
    '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
    # 使用np.dot进行矩阵乘法，得到降维后的矩阵Z 
    Z = np.dot(self.X, self.U)  
    # 输出X的形状，U的形状，以及Z的形状，用于检查是否符合预期 
    print('X shape:', np.shape(self.X))
    print('U shape:', np.shape(self.U))
    print('Z shape:', np.shape(Z))
    # 打印降维后的矩阵Z 
    print('样本矩阵X的降维矩阵Z:\n', Z)
    # 返回降维后的矩阵Z 
    return Z 

在这个方法中，np.dot(self.X, self.U) 这行代码执行了矩阵乘法，即将每个样本点 X[i] 乘以转换矩阵 U，从而得到降维后的样本点 Z[i]。
由于 U 是一个 (n, k) 形状的矩阵，其中 n 是原始特征维度，k 是降维后的特征维度，所以 Z 的形状将是 (m, k)，其中 m 是样本数。

在实践中，您可能会希望省略掉打印语句，特别是在处理大型数据集时，因为这些打印语句会影响性能，并且可能不适合在生产环境中使用。
此外，如果您正在使用 Python 3.8 或更高版本，您可以使用 f-string 来格式化字符串，使代码更加整洁和易读。例如：

print(f'X shape: {np.shape(self.X)}')
print(f'U shape: {np.shape(self.U)}')
print(f'Z shape: {np.shape(Z)}')
    """


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

    """
    if __name__ == '__main__':
    # 创建一个10样本3特征的样本集 
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
    # K是降维后的特征维度，这里等于样本集的列数减1 
    K = np.shape(X)[1] - 1 
    # 打印原始样本集 
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    # 初始化CPCA对象 
    pca = CPCA(X, K)
    """