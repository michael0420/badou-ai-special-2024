'''
PCA手算降维，简明版
'''

import numpy as np

class PCA(object):
    def __init__(self, dimension):  # 即输入的参数
        self.dimension = dimension

    def convert(self, X):
        self.X = X
        m, n = self.X.shape  # 样本的特征维度
        Avarage_X = self.X - self.X.mean(axis=0)  # 中心化处理
        # self.X = self.X - np.array([np.mean(self.X, axis=0)])
        Covariance_X = np.dot(Avarage_X.T, Avarage_X) / (m - 1)
        eigenvalues, eigenvectors = np.linalg.eig(Covariance_X)
        index = np.argsort(-1 * eigenvalues)
        base_vectors = np.transpose([eigenvectors[:, index[i]] for i in range(self.dimension)])
        new_X = np.dot(self.X, base_vectors)
        print(base_vectors)
        return new_X


if __name__ == '__main__':
    A = np.array([[10, 15, 29],
                 [15, 46, 13],
                 [23, 21, 30],
                 [11, 9,  35],
                 [42, 45, 11],
                 [9,  48, 5],
                 [11, 21, 14],
                 [8,  5,  15],
                 [11, 12, 21],
                 [21, 20, 25]])
    pca = PCA(2)
    B = pca.convert(A)
    print('样本矩阵降维后矩阵为：\n', B)

# import numpy as np
#
#
# class PCA(object):
#     def __init__(self, X, K):
#         self.X, self.K = X, K
#         X_new = self.convert()
#
#     def convert(self):
#         m, n = self.X.shape
#         mu = np.array([np.mean(self.X, axis=0)])
#         X_center = self.X - mu
#         X_center_cov = 1 / (m - 1) * np.dot(X_center.T, X_center)
#         eigenvalues, eigenvectors = np.linalg.eig(X_center_cov)
#         index = np.argsort(-1 * eigenvalues)
#         temp = np.array([eigenvectors[:, index[i]] for i in range(self.K)])
#         X_base_vectors = temp.T
#         X_new = np.dot(self.X, X_base_vectors)
#         print('降至%d维的数据矩阵为：\n' % self.K, X_new)
#         print(X_base_vectors)
#         return X_new
#
#
# if __name__ == '__main__':
#     A = np.array([[10, 15, 29],
#                  [15, 46, 13],
#                  [23, 21, 30],
#                  [11, 9,  35],
#                  [42, 45, 11],
#                  [9,  48, 5],
#                  [11, 21, 14],
#                  [8,  5,  15],
#                  [11, 12, 21],
#                  [21, 20, 25]])
#     PCA(A, 2)