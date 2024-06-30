"""
相关特征
无关特征
冗余特征

PCA实现步骤：
1. 中心化，或者叫零均值化
2. 求协方差矩阵
3. 对协方差矩阵求 【特征向量】和【特征值】，特征向量组成了新的特征空间

PCA降维的意义：
1.降维后同一维度的方差最大，即坐标点越分散 越能更好表示源数据
2.不同维度的相关性为0


眼 鼻 嘴   |  眼 鼻 嘴  |  眼 鼻 嘴
大 小 大   |  大 大 大  |  大 中 小
中 小 中   |  中 中 中  |  中 小 大
小 小 小   |  小 小 小  |  小 大 中

第一个都是小鼻子，则鼻子是无关变量，不满足同一维度方差大
第二个虽然是同一维度方差最大，鼻子各不相同，但是不同维度相关性强，例如可以从大眼推出大鼻和大嘴
第三个则同时满足同一维度方差最大，不同维度相关性低，要的是这种

协方差是用来表示两个随机变量关系的统计量
例如 [x,y,z]， 则要求[x,y], [x,z], [y,z]的相关性

"""

import numpy as np
from sklearn.decomposition import PCA


class MyPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        # 每行是一个样本，每列是一个维度
        # axis = 1会计算每一行的平均值
        # axis = 0会计算每一列的平均值
        # 零均值化
        X = X - X.mean(axis=0)
        print(X.mean(axis=0))

        # 求协方差矩阵
        cov_matrix = np.dot(X.T, X) / X.shape[0]  # X.T表示矩阵的转置
        print("协方差矩阵\n", cov_matrix)

        # 协方差矩阵的特征值，特征向量
        eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
        print("特征值\n", eig_vals)
        print("特征向量\n", eig_vectors)

        # 特征值降序排列的序号
        idx = np.argsort(-eig_vals)
        print(type(idx))
        print("特征值降序排列下标\n", idx)

        # 降维特征向量矩阵
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        print("降维特征向量矩阵\n", self.components_)

        # 对X进行降维
        return np.dot(X, self. components_)


def pca_manual():
    # 生成10*4随机整数矩阵，整数值在0到100之间
    old_matrix = np.random.randint(0, 100, (6, 4))
    print("原矩阵\n", old_matrix)
    my_pca = MyPCA(2)
    new_matrix = my_pca.fit_transform(old_matrix)
    print("=" * 10)
    print("新矩阵\n", new_matrix)


def pca_sklearn():
    old_matrix = np.random.randint(0, 100, (5, 4))
    print("原矩阵\n", old_matrix)
    pca = PCA(n_components=3)
    # pca.fit(old_matrix)
    new_matrix = pca.fit_transform(old_matrix)
    print("=" * 10)
    print("新矩阵\n", new_matrix)


if __name__ == '__main__':
    # pca_manual()
    pca_sklearn()

