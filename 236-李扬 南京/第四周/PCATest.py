import numpy as np

class PCA():
    def __init__(self, component):
        self.component = component

    #X为样本矩阵
    def PCAA(self, X):
        #去中心化
        X = X - X.mean(axis=0)
        #求协方差矩阵
        self.X = np.dot(X.T, X)/X.shape[0]
        #求特征值和特征向量
        vals, vectors = np.linalg.eig(self.X)
        #进行降序，并获取索引
        indexList = np.argsort(-1 * vals)
        #降维矩阵
        components = vectors[:, indexList[:self.component]]
        #乘积获取新矩阵
        return np.dot(X, components)

pca3 = PCA(3)
pca2 = PCA(2)
#测试用数据
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
print(X)
newX3= pca3.PCAA(X)
newX2 = pca2.PCAA(X)
print(newX3)
print(newX2)