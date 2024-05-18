import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt




class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        # 计算均值
        mean_X = np.mean(X, axis=0)
        print("均值：\n", mean_X)
        # 中心化
        X = X - mean_X
        print("中心化后的数据：\n", X)

        # 计算协方差矩阵
        cov_x = np.dot(X.T, X) / X.shape[0]
        print("协方差矩阵：\n", cov_x)

        # 计算特征值和特征向量
        value, vector = np.linalg.eig(cov_x)
        print("特征值：\n", value)
        print("特征向量：\n", vector)

        # 按照特征值升序排序
        index = np.argsort(-value)
        value = value[index]
        vector = vector[:, index]
        print("按照特征值升序排序后：\n", value)
        print("按照特征值升序排序后的特征向量：\n", vector)

        # 选取前两维作为主成分
        vector = vector[:, :self.n_components]
        # 计算主成分分析后的数据
        pca_X = X.dot(vector)
        print("主成分分析后的数据：\n", pca_X)
        return pca_X


# 随机生成三维15列的数据集作为PCA样本
# X = np.array([[9, 2, 3],
#               [3, 5, 2],
#               [12, 4, 6],
#               [7, 8, 1],
#               [1, 10, 9],
#               [4, 6, 8],
#               [11, 12, 13],
#               [5, 7, 14],
#               [2, 11, 15],
#               [8, 13, 10],
#               ])
# pca = PCA(n_components=2)
# pca.fit(X)

# 尝试对鸢尾花数据集进行PCA分析
X, y = load_iris(return_X_y=True)
pca = PCA(n_components=2)
pca_X = pca.fit(X)

# 打印结果
red_X, red_y = [], []
blue_X, blue_y = [], []
green_X, green_y = [], []
for i in range(len(pca_X)):
    if y[i] == 0:
        red_X.append(pca_X[i][0])
        red_y.append(pca_X[i][1])
    elif y[i] == 1:
        blue_X.append(pca_X[i][0])
        blue_y.append(pca_X[i][1])
    else:
        green_X.append(pca_X[i][0])
        green_y.append(pca_X[i][1])
plt.scatter(red_X, red_y, c='r', marker='X')
plt.scatter(blue_X, blue_y, c='b', marker='D')
plt.scatter(green_X, green_y, c='g', marker='.')
plt.show()
