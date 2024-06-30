import numpy as np
from sklearn.decomposition import PCA
'''
使用PCA求样本矩阵X的K阶降维矩阵Z

PCA算法的原理：
将数据从原始的空间中转换到新的特征空间中
eg:原特征空间为：(x, y, z) --> 转换后新的特征空间为(a, b, c)
   但是在新的特征空间中 a,b,c为新的基
   所有数据在c上的投影都接近于0,则可以忽略，特征空间可以降维成二维的(a, b)
   
PCA算法的步骤：
1.对原始数据零均值化(中心化)：样本矩阵(输入)记作X 中心化后的样本矩阵为centrX
2.求协方差矩阵C: 中心化后可由公式求得协方差矩阵 C = 1/m * (CentrX^T · CentrX) m为样本个数
3.对协方差矩阵求特征值a和特征向量b，将a按从大到小顺序排列，取对应前K个b中的特征向量组成降维转换矩阵U(新的特征空间)
4.降维后的样本矩阵Z由Z = XU求出
'''

class diy_PCA(object):

    def __init__(self, X, K):
        # X: X为样本矩阵，保证输入的X.shape = (m, n) m行样例 n个特征
        # K: X的降维矩阵的阶数，即要将特征值矩阵X降阶到K维

        self.X = X # 样本矩阵X
        self.K = K # K阶降维矩阵的K值
        self.centrX = [] # 矩阵X中心化后的结果(中心化矩阵)
        self.C = [] # 样本集的协方差矩阵C
        self.U = [] # 样本矩阵X的降维转换矩阵U(即在求得的特征向量矩阵选取前K列得到的矩阵)
        self.Z = [] # 样本矩阵X的降维矩阵Z Z = XU

        # 运算
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        # 矩阵X中心化
        print("------样本中心化------")
        print("样本矩阵X:\n", self.X)
        centrX = [] # 初始化中心矩阵
        mean = np.array([np.mean(attr) for attr in self.X.T]) # 样本集的特征均值
        # for attr in self.X.T中 attr会遍历self.X.T的每一行，即每次循环attr都被赋值为self.X.T的下一行
        # 即attr是每个特征下的所有数据的列表(即X的每一列)
        # 对其取均值np.mean(attr)可获得每个特征下所有数据的均值(X每一列的均值)
        print("样本集的特征均值:\n",mean)
        centrX = self.X - mean # 样本集的中心化 每个样本的特征值都减去当前类的均值
        print("样本矩阵X的中心化结果centrX:\n", centrX)
        return centrX

    def _cov(self):
        # 求样本矩阵X的协方差矩阵C
        # 样本集的样例总数
        ns = np.shape(self.centrX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)
        # 中心化后可由公式求得协方差矩阵 C = 1/m * (CentrX^T · CentrX) m为样本个数
        '''
        ↑ 均值已经用了n个数的平均来做估计 在求方差时，只有(n-1)个数和均值信息是不相关的。
        第n个样本可由前(n-1)个样本和均值来唯一确定，实际上没有信息量。
        所以在计算方差时，只除以(n-1)。
        '''
        print("样本矩阵X的协方差矩阵C:\n", C)
        return C

    def _U(self):
        # 求X的降维转换矩阵U， shape = (n, k), n是X的特征维度总数， k是降维矩阵的的特征维度
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C) # 特征值赋值给a，对应特征向量赋值给b。
        # 函数doc：https: // docs.scipy.org / doc / numpy - 1.10.0 / reference / generated / numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        # 给出特征值降序的topK索引序列
        ind = np.argsort(-1 * a) # 调用会返回一个索引数组 元素按照-1*a（即a的元素取负）的升序排列
        # 相当于对原特征值数组进行降序排列
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:, ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U) # 注意这里是用的原始样本矩阵和U点乘 而在sklearn中是用中心化矩阵点乘的
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

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
    diy_pca = diy_PCA(X,K)

    '''使用sklearn库调用PCA'''
    sklearn_pca = PCA(n_components= 2) # 降到2维
    sklearn_pca.fit(X) # 执行
    newX = sklearn_pca.fit_transform(X) # 降维后的数据
    print("sklearn PCA执行结果:\n", newX)