
import numpy as np

class PCA(object):

    #样本矩阵X shape=(m,n) , m行样例，n个特征
    def __init__(self,X,K):
        self.X = X   #样本矩阵
        self.K = K   #降维后的矩阵维数

        self.centra_X = self._centra_X()  #矩阵X的中心化
        self.cov = self._cov()   #协方差矩阵
        self.U = self._U()    #降维转换矩阵
        self.Z = self._Z()    #Z=XU:降维后的新矩阵

        # 矩阵X的中心化
    def _centra_X(self):


        mean = np.array([np.mean(attribute) for attribute in self.X.T])  #样本特质的均值
        centra_X = self.X - mean     #样本中心化

        print('样本矩阵X：\n', self.X)
        print('样本集的特征均值为：\n',mean)
        print('样本矩阵X的中心化：\n',centra_X)
        return centra_X


        # 矩阵X的协方差矩阵cov
    def _cov(self):
        sum_X =  np.shape(self.centra_X)[0]
        cov = np.dot(self.centra_X.T,self.centra_X)/(sum_X - 1)
        print('样本矩阵的协方差矩阵C:\n',cov)
        return cov

        # X的减幅为转换矩阵U（特征值向量从大到小）
    def _U(self):

        a,b = np.linalg.eig(self.cov)  #求特征值、特征向量。a为特征值，b为特征向量。
        index = np.argsort(-1*a)  #特征值降序是我索引序列
        UT = [b[:,index[i]] for i in range(self.K)]  #K阶降维的转换矩阵
        U = np.transpose(UT)
        print('样本集的协方差矩阵cov的特征值:\n', a)
        print('样本集的协方差矩阵cov的特征向量:\n', b)
        print('%d阶降维转换矩阵U:\n' % self.K, U)

        #评价模型好坏
        a1 = a[index[:K]]
        percent = np.sum(a1) / np.sum(a)
        print('评价结果为：', percent)



        return U

        # 按照Z=XU求降维矩阵Z
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__=='__main__':
    X = np.array([[14,16,31],
                  [16,34,88],
                  [22,33,44],
                  [18,25,17],
                  [28,4,31],
                  [2,44,55],
                  [31,35,13],
                  [24,6,15],
                  [27,25,26],
                  [19,12,11]])

    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X,K)
