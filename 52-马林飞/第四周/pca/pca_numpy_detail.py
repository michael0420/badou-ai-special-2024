import numpy as np


class PCA:
    def __init__(self, sample_matrix, dim):
        self.sample_matrix = sample_matrix  # 样本矩阵
        self.dim = dim  # 需要到达的维度，即降维后的维度数
        self.center_matrix = []  # 中心化矩阵，除与矩阵后的矩阵
        self.cov_matrix = []  # 协方差矩阵
        self.feature_matrix = []  # 通过协方差矩阵特征值和特征向量，由大到小矩阵的特征向量矩阵
        self.dim_reduction_matrix = []  # 降维后的样本矩阵

    def _centralized(self):
        sample_matrix_col_mean = np.mean(self.sample_matrix, axis=0)
        self.center_matrix = self.sample_matrix - sample_matrix_col_mean
        print('样本矩阵X的中心化center_matrix:\n', self.center_matrix)

    def _cov(self):
        size = np.shape(self.center_matrix)[0]
        self.cov_matrix = np.dot(self.center_matrix.T, self.center_matrix) / size
        print('样本矩阵X的协方差矩阵cov_matrix:\n', self.cov_matrix)

    def _feature(self):
        feature_value, feature_vector = np.linalg.eig(self.cov_matrix)
        print('样本集的协方差矩阵C的特征值:\n', feature_value)
        print('样本集的协方差矩阵C的特征向量:\n', feature_vector)

        des_feature_value = np.argsort(feature_value * -1)

        feature_matrix_T = [feature_vector[:, des_feature_value[i]] for i in range(self.dim)]
        self.feature_matrix = np.transpose(feature_matrix_T)

        print('%d阶降维转换矩阵feature_matrix:\n' % self.dim, self.feature_matrix)

    def _dim_matrix(self):
        self.dim_reduction_matrix = np.dot(self.center_matrix, self.feature_matrix)

        print('样本矩阵X的降维矩阵dim_reduction_matrix:\n', self.dim_reduction_matrix)


if __name__ == '__main__':
    matrix_1 = np.array([[10, 15, 29],
                         [15, 46, 13],
                         [23, 21, 30],
                         [11, 9, 35],
                         [42, 45, 11],
                         [9, 48, 5],
                         [11, 21, 14],
                         [8, 5, 15],
                         [11, 12, 21],
                         [21, 20, 25]])
dim = np.shape(matrix_1)[1] - 1

pca = PCA(matrix_1, dim)

pca._centralized()
pca._cov()
pca._feature()
pca._dim_matrix()
