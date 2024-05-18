import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class CPCA:
    '''利用PCA方法降低样本矩阵X的维度至K阶。
    注意：确保输入的样本矩阵X的形状为(m, n)，其中m为样本数，n为特征数。
    '''
    def __init__(self, X, K):
        '''
        参数:
        X : ndarray
            样本矩阵X。
        K : int
            降维后的维度数。
        '''
        self.X = np.array(X)  # 将样本矩阵X转换为ndarray
        self.K = K  # 降维矩阵的维度
        self.centralized_X = self._centralize()
        self.cov_matrix = self._calculate_covariance()
        self.transformation_matrix = self._determine_transformation_matrix()
        self.Z = self._calculate_Z()  # 通过Z=XU得到降维矩阵

    def _centralize(self):
        '''中心化矩阵X。'''
        # print('样本矩阵X:\n', self.X)
        # mean = np.array([np.mean(attr) for attr in self.X.T])  # 样本集的特征均值
        mean = np.mean(self.X, axis=0)  # 计算特征的均值
        # print('样本集的特征均值:\n', mean)
        centralized_X = self.X - mean
        # print('样本矩阵X的中心化centrX:\n', centrX)
        return centralized_X

    def _calculate_covariance(self):
        '''计算样本矩阵X的协方差矩阵C。'''
        num_samples = self.centralized_X.shape[0]
        covariance_matrix = np.dot(self.centralized_X.T, self.centralized_X) / (num_samples - 1)
        return covariance_matrix

    def _determine_transformation_matrix(self):
        '''求解降维转换矩阵U，形状为(n, k)，n为特征总数，k为降维后的特征维度。'''
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_matrix)
        # print('样本集的协方差矩阵C的特征值:\n', eigenvalues)
        # print('样本集的协方差矩阵C的特征向量:\n', eigenvectors)
        # 给出特征值降序的topK的索引序列
        sorted_indices = np.argsort(-eigenvalues)
        # 构建K阶降维的降维转换矩阵U
        transformation_matrix = eigenvectors[:, sorted_indices[:self.K]]
        return transformation_matrix

    def _calculate_Z(self):
        '''计算降维矩阵Z，形状为(m, k)，m为样本数，k为降维后的特征维度。'''
        Z = np.dot(self.X, self.transformation_matrix)
        return Z

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # 数据中心化
        X_centered = X - np.mean(X, axis=0)
        # 计算协方差矩阵
        covariance_matrix = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
        # 特征值和特征向量
        eig_vals, eig_vectors = np.linalg.eig(covariance_matrix)
        idx = np.argsort(-eig_vals)
        components = eig_vectors[:, idx[:self.n_components]]
        # 数据投影到主成分
        return np.dot(X_centered, components)

    def plot_pca(self, reduced_data, labels):
        # 准备绘图
        colors = ['r', 'b', 'g']  # 不同类别的颜色
        markers = ['x', 'D', '.']  # 不同类别的标记
        categories = np.unique(labels)  # 标签的唯一类别

        plt.figure(figsize=(10, 7))
        for category, color, marker in zip(categories, colors, markers):
            # 根据类别过滤数据
            cat_data = reduced_data[labels == category]
            plt.scatter(cat_data[:, 0], cat_data[:, 1], c=color, marker=marker, label=str(category))

        plt.title('PCA Dimensionality Reduction Visualization')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='best')
        plt.show()

if __name__ == '__main__':
    data = pd.read_csv('normalized_data.csv')
    features = data.drop('Survived', axis=1)  # 确保这里正确地去除了标签列
    labels = data['Survived']  # 假设标签列名为 'Survived'
    pca = PCA(2)
    reduced_data = pca.fit_transform(features.values)
    pca.plot_pca(reduced_data, labels)  # 使用 PCA 类的绘图方法