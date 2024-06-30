import numpy as np

class PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        #X矩阵中心化
        centerX = X - X.mean(axis=0)
        #求协方差矩阵
        self.C = np.dot(centerX.T,centerX)/centerX.shape[1]
        #求协方差矩阵的特征值和特征向量
        a,b = np.linalg.eig(self.C)
        #求a从大到小排序后的索引
        index = np.argsort(-a)
        #求前n个大的特征值对应的特征向量
        self.U = b[:, index[:self.n_components]]
        #求降维后的矩阵
        self.Z = np.dot(centerX, self.U)
        return (self.Z)

pca = PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
newX = pca.fit_transform(X)
print(newX)


