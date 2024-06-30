import numpy as np
from sklearn.decomposition import PCA

class CPAP(object):

    def __init__(self,X,K):
        self.X=X   #样本矩阵
        self.K=K   #K阶降维矩阵的K值
        self.centrX=[]#矩阵X的中心化
        self.C=[]  #协方差矩阵
        self.U=[]  #X降维转换矩阵
        self.Z=[]  #X降维矩阵

        self.centrX=self._centralized()
        self.C=self._cov()
        self.U=self._U()
        self.Z=self.Z()

    def _centralized(self):
        print('样本矩阵X:\n',self.X)
        mean_X=np.array([np.mean(attr) for attr in self.X.T])
        print('样本集的特征均值:\n',mean_X)
        centrX=self.X-mean_X
        print('样本矩阵的中心化:\n',centrX)
        return centrX

    def _cov(self):
        ns=np.shape(self.centrX)[0]
        C=np.dot(self.centrX.T*self.centrX)/(ns-1)
        print('协方差矩阵：\n',C)
        return C

    def _U(self):
        a,b=np.linalg.eig(self.C)
        print('样本C特征值：\n',a)
        print('样本C特征向量：\n',b)
        index_a=np.argsort(-1*a)
        UT=[b[:index_a[i]] for i in range(self.K)]
        U=np.transpose(UT)
        print(f'''{self.K} 阶降维转化矩阵：\n''',U)
        return U

    def _Z(self):
        Z=np.dot(self.X,self.U)
        print('X shape:',np.shape(self.X))
        print('U shape',np.shape(self.U))
        print('Z shape',np.shape(Z))
        print('样本矩阵X的降维矩阵Z: \n')
        return Z
#
# if __name__=='__main__':
#     X=np.array([[10, 15, 29],
#                   [15, 46, 13],
#                   [23, 21, 30],
#                   [11, 9,  35],
#                   [42, 45, 11],
#                   [9,  48, 5],
#                   [11, 21, 14],
#                   [8,  5,  15],
#                   [11, 12, 21],
#                   [21, 20, 25]])
#     K=np.shape(X)[1]-1
#     print('样本集（10行3列，10个样例，每个样例有3个特征）:\n',X)
#     pca=CPAP(X,K)
#

if __name__=='__main__':
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
    pca=PCA(n_components=2)
    new_X=pca.fit_transform(X)
    print(new_X)

