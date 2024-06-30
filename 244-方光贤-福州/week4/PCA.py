import numpy as np

class PCA():
    def __init__(self, n_component):
        #设置pca类属性 即维度
        self.n_component = n_component

    def fit_transform(self, X):
        #特征数等于X的列数
        self.n_features_ = X.shape[1]
        #中心化 求协方差矩阵
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        #求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        #降序排列特征值并得到相关序号
        idx = np.argsort(-eig_vals)
        #对矩阵降维
        self.component_ = eig_vectors[:, idx[:self.n_component]]
        #对X进行降维
        return np.dot(X, self.component_)

#调用pca方法
pca = PCA(n_component = 2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
newX=pca.fit_transform(X)
print(newX)                  #输出降维后的数据

