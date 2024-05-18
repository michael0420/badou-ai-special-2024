import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

# 鸢尾花实例
# 加载数据，x表示数据集中的属性数据，y表示数据标签
# x（或通常命名为X，以符合惯例）代表数据集的特征矩阵。
# 对于鸢尾花数据集，每一行代表一个样本（即一朵花），
# 每一列代表一个特征（如花萼长度、花萼宽度、花瓣长度、
# 花瓣宽度）。因此，X是一个形状为(n_samples, n_features)
# 的二维数组，其中n_samples是样本数量（通常是150），n_features
# 是特征数量（对于鸢尾花数据集是4）。
#
# y代表与每个样本对应的标签向量。对于鸢尾花数据集，标签
# 表示每种花的种类，通常是一个长度为n_samples的一维数组，
# 其中的值可以是0、1或2，分别对应山鸢尾、变色鸢尾和维吉尼亚鸢尾。
x, y = load_iris(return_X_y=True)
# 加载PCA算法，设置降维后主成分数量为2（即降维到二维）
pca = dp.PCA(n_components=2)
# 对原始数据进行降维，保存在reduced_x中
reduced_x = pca.fit_transform(x)
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_x)):  # 遍历降维后的数据点
    if y[i] == 0:  # 如果当前数据点的标签是0（代表山鸢尾）
        red_x.append(reduced_x[i][0])  # 将该数据点的第一个维度（x坐标）添加到red_x列表中
        red_y.append(reduced_x[i][1])  # 将该数据点的第二个维度（y坐标）添加到red_y列表中
    elif y[i] == 1:  # 如果当前数据点的标签是1（代表变色鸢尾）
        blue_x.append(reduced_x[i][0])  # 将该数据点的第一个维度（x坐标）添加到blue_x列表中
        blue_y.append(reduced_x[i][1])  # 将该数据点的第二个维度（y坐标）添加到blue_y列表中
    else:  # 如果当前数据点的标签是2（代表维吉尼亚鸢尾）
        green_x.append(reduced_x[i][0])  # 将该数据点的第一个维度（x坐标）添加到green_x列表中
        green_y.append(reduced_x[i][1])  # 将该数据点的第二个维度（y坐标）添加到green_y列表中

# c参数代表图像的颜色，marker代表图像的形状
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()


# 自主实现PCA算法，简化版
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X: np.ndarray):
        # 中心化
        # 进行矩阵（或数组）X的中心化（centering）。具体来说，X是一个二维
        # 数组（通常表示数据集中的多个样本和特征），而X.mean(axis=0)计算了
        # X沿着第一个轴（即列，对应特征）的平均值。X - X.mean(axis=0)则从
        # X中的每一列中减去该列的平均值，这样得到的新X矩阵的每一列的平均值都为0。
        X = X - X.mean(axis=0)
        # 求协方差矩阵，注意，这里的X.T和X的顺序不能调换，因为矩阵乘不满足交换律
        covariance_matrix = np.dot(X.T, X) / X.shape[0]
        # 通过协方差矩阵求特征值和特征向量
        eigen_value, eigen_vector = np.linalg.eig(covariance_matrix)
        # 对特征值进行降序排序
        sorted_indices = np.argsort(eigen_value)[::-1]
        # 通过特征值对应的前n_components和特征向量求出降维特征矩阵(也叫降维转换矩阵)
        top_k_eigenvectors = eigen_vector[:, sorted_indices[:self.n_components]]
        # 原矩阵点乘降维特征矩阵就可以把原矩阵降维成与特征矩阵具有相同维数的矩阵了
        X = np.dot(X, top_k_eigenvectors)
        return X


# 调用
pca = PCA(n_components=2)
X = np.array(
    [[-1, 2, 66, -1],
     [-2, 6, 58, -1],
     [-3, 8, 45, -2],
     [1, 9, 36, 1],
     [2, 10, 62, 1],
     [3, 5, 83, 2]])  # 导入数据，维度为4
newX = pca.fit_transform(X)
print(newX)  # 输出降维后的数据


# 自主实现PCA算法，详细版（封装）
class CPCA(object):

    def __init__(self, X: np.ndarray, K):
        self.X = X  # 样本矩阵
        self.K = K  # 要降到的目标维数
        self.centrX = []  # 中心化之后的矩阵
        self.covariance_matrix = []  # 协方差矩阵
        self.eigenvectors = []  # 降维特征矩阵（也叫降维转换矩阵）
        self.Z = []  # 样本矩阵X的降维矩阵Z(PCA的结果矩阵)

        self.centrX = self._Centralized()
        self.covariance_matrix = self._Cov()
        self.eigenvectors = self._Eig()
        self.Z = self._Dim()

        # self._Centralized()
        # self._Cov()
        # self._Eig()
        # self._Dim()
        # return Z

    def _Centralized(self):
        # 中心化
        centrX = self.X - self.X.mean(axis=0)
        return centrX

    def _Cov(self):
        # 求协方差矩阵
        C = np.dot(self.centrX.T, self.centrX) / self.centrX.shape[0]
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _Eig(self):
        # 求降维特征矩阵（也叫降维转换矩阵）
        eigen_value, eigen_vector = np.linalg.eig(self.covariance_matrix)
        # 排序后返回的是特征值数组的索引数组，并不是特征值数组本身的值
        ind = np.argsort(-1 * eigen_value)
        # 取出特征向量的前K列，形成降维矩阵
        # eigen_vector[:, ind[:self.K]]: 这是使用NumPy风格的切片来从
        # eigen_vector中提取特定的列。: 表示选择所有行，而ind[:self.K]
        # 表示选择前self.K个索引对应的列。
        print("--------------------------------")
        print(eigen_vector)
        E = eigen_vector[:, ind[:self.K]]
        print(E)
        print("--------------------------------")
        print('样本集的协方差矩阵C的特征值:\n', eigen_value)
        print('样本集的协方差矩阵C的特征向量:\n', eigen_vector)
        print('%d阶降维转换矩阵E:\n' % self.K, E)
        return E

    def _Dim(self):
        # 求出降维之后的矩阵
        Z = np.dot(self.centrX, self.eigenvectors)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.covariance_matrix))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X, K)


# 调用sklearn.datasets中的方法进行PCA主成分分析
# X = np.array(
#     [[-1, 2, 66, -1],
#      [-2, 6, 58, -1],
#      [-3, 8, 45, -2],
#      [1, 9, 36, 1],
#      [2, 10, 62, 1],
#      [3, 5, 83, 2]])  # 导入数据，维度为4
X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
pca = PCA(n_components=2)
newX = pca.fit_transform(X)
print(newX)
