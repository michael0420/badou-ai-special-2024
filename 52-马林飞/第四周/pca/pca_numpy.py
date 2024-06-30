import numpy as np


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        # 求协方差矩阵
        X = X - X.mean(axis=0)
        print('样本矩阵X的中心化X:\n', X)

        self.covariance = np.dot(X.T, X) / X.shape[0]
        print('样本矩阵X的协方差矩阵covariance:\n', self.covariance)

        # 求协方差矩阵的特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        print('样本集的协方差矩阵C的特征值:\n', eig_vals)
        print('样本集的协方差矩阵C的特征向量:\n', eig_vectors)

        # 获得降序排列特征值的序号
        idx = np.argsort(-eig_vals)
        # 降维矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        print('%d阶降维转换矩阵components_:\n' % self.n_components, self.components_)

        # 对X进行降维
        print(X)
        print(self.components_)

        return np.dot(X, self.components_)


# 调用
pca = PCA(n_components=2)
X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])  # 导入数据，维度为3
newX = pca.fit_transform(X)
print('-' * 8)
print(newX)  # 输出降维后的数据
