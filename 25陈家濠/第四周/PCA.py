import numpy as np
class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
    #给实例绑定属性
    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        #x的列数也就是所谓的特征数量
        X = X - X.mean(axis=0)
        #中心化
        self.covariance = np.dot(X.T, X) / X.shape[0]
        #协方差C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        # 特征值赋值给1，对应特征向量赋值给2。
        idx = np.argsort(-eig_vals)
        #特征向量降序
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        #截取需要的降维范围
        return np.dot(X, self.components_)
pca = PCA(n_components=2)
#把component设定为2，之后输入的都降到二维
X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
newX = pca.fit_transform(X)
print(newX)