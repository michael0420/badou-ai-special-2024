import numpy as np

'''用PCA求样本矩阵X    计算出K阶降维后的矩阵Z
Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
'''
class CPCA(object):
    def __init__(self, X, K):  # 初始化方法, X为样本矩阵X ,K为X要降维成k阶
        self.X = X  # 样本矩阵X
        self.K = K  # 保留的维度数
        self.centrX = []  # 矩阵X的中心化后的矩阵
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 特征向量矩阵
        self.Z = []  # 降维后的结果

        self.centrX = self._centralized()  # 中心化调用
        self.C = self._cov()  # 计算协方差矩阵
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):  # 中心化方法
        print('样本矩阵X:\n', self.X)
        mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值     .T: 这是NumPy数组的一个属性，表示其转置。
        print('样本集的特征均值:\n', mean)
        centrX = self.X - mean  # 样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX

    def _cov(self):  # 求样本矩阵X的协方差矩阵C
        ns = np.shape(self.centrX)[0]     # 原始特征的行数
        C = np.dot(self.centrX.T, self.centrX) / (ns - 1)  # 样本矩阵的协方差矩阵C,这里矩阵乘的时候，转置矩阵一定要放在前面，这样结果则是输入矩阵的特征列数的方阵。
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):  # 求特征向量矩阵       求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度
        # 先求X的协方差矩阵C的特征值和特征向量
        a, b = np.linalg.eig(self.C)  # 特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)

        ind = np.argsort(-1 * a)  # 给出特征值降序的topK的  索引序列
        UT = [b[:, ind[i]] for i in range(self.K)]  # 构建K阶降维的降维转换矩阵UT
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):  # 按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数
        Z = np.dot(self.X, self.U)  # 在吴恩达的机器学习课中，这里是用U的转置放前面， 然后乘以每一列数据， 得到降维后的每一列数据， 见吴恩达14-4
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


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
print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
pca = CPCA(X, 2)
