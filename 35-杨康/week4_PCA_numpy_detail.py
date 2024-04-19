"""
使用PCA求样本矩阵X的降维矩阵Z
1. 对原始数据零均值化（中心化）centerX，
2. 求协方差矩阵， C=1/m * centerXTcenterX  m为样本个数(行数)
3. 对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间。按特征值大小排序前k个对应的特征向量组成U
4. Z=XU
"""
import numpy as np

class PCA():
    def __init__(self,X,K):
        self.X = X                    #样本矩阵X
        self.K = K                    #K阶降维矩阵的K值
        self.centerX = self._center() #中心化矩阵(零均值化)
        self.C = self._cov()          #协方差矩阵 C=1/m * centerXTcenterX  m为样本个数(行数)
        self.U = self._U()            #取协方差矩阵的前K维特征向量作为X的降维转换矩阵
        self.Z = self._Z()            #Z=XU

    def _center(self):
        print('样本矩阵：\n', self.X)
        m = self.X.mean(axis=0)   #求X各特征均值（列）
        print('特征均值：\n', m)
        centerX = self.X-m
        print('中心化矩阵：\n', centerX)
        return centerX

    def _cov(self):
        C = np.dot(self.centerX.T, self.centerX)/self.centerX.shape[0]   #协方差矩阵 C=1/m * centerXTcenterX  m为样本个数(行数)
        print('协方差矩阵：\n', C)
        return C

    def _U(self):
        # 求协方差矩阵的特征值a，特征向量b
        a,b = np.linalg.eig(self.C)        #特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html
        print('特征值:\n', a)
        print('特征向量:\n', b)
        # 降序后的特征值索引
        index = np.argsort(-1*a)           #np.argsort() 从小到大排序后输出对应的索引
        UT = [b[:,index[i]] for i in range(self.K)]
        U = np.transpose(UT)
        # U = b[:,index[0:self.K]]
        print('X的%d阶降维转换矩阵：\n' % self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', self.X.shape)
        print('U shape:', self.U.shape)
        print('Z shape:', Z.shape)
        print('降维后矩阵Z：\n', Z)
        return Z

if __name__ == "__main__":
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
    K = X.shape[1]-1
    pca = PCA(X, K)

