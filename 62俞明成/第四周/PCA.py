import numpy as np


class PCA:
    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.center_x = self.centralized()
        self.c = self.cov()
        self.u = self.u()
        self.z = self.z()

    def centralized(self):
        # 每个特征的均值
        mean = np.array([np.mean(i) for i in self.x.T])
        # print(mean)
        # 中心化
        center_x = self.x - mean
        # print(center_x)
        return center_x

    def cov(self):
        # 样本数量
        num = self.x.shape[0]
        # print(num)
        # 协方差矩阵
        D = np.dot(self.x.T, self.x) / (num - 1)
        # print(D)
        return D

    def u(self):
        # a特征值，b特征向量
        a, b = np.linalg.eig(self.c)
        print(a)
        print(b)
        # 对特征值进行从大到小排序
        sort_a = np.argsort(-1 * a)
        # print(sort_a)
        # 选择其中最大的k个，然后将其对应的k个特征向量分别作为列向量
        # 组成特征向量矩阵W_nxk
        ut = [b[:, sort_a[i]] for i in range(self.k)]
        u = np.transpose(ut)
        # print("X--shape", np.shape(self.x))
        # print("U--shape", np.shape(u))
        return u

    def z(self):
        # 计算X_new*W
        return np.dot(self.x, self.u)


if __name__ == '__main__':
    data = np.array([[10, 15, 29],
                     [15, 46, 13],
                     [23, 21, 30],
                     [11, 9, 35],
                     [42, 45, 11],
                     [9, 48, 5],
                     [11, 21, 14],
                     [8, 5, 15],
                     [11, 12, 21],
                     [21, 20, 25]])
    K = data.shape[1] - 1
    pca = PCA(data, K)
    B = np.array([[1, 2, 4], [1, 4, 5]])
    print(B.shape)
