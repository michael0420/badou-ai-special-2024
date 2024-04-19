import numpy as np


class PCA(object):
    """
    PCA简化版实现
    """

    def __init__(self, X, k, is_print=True):
        self.X = X
        self.k = k

    def to_PCA(self):
        # 中心化
        X = self.X - self.X.mean(axis=0)
        print("中心化", X)
        # 协方差
        xfc = np.dot(X.T, X) / (X.shape[0] - 1)
        print("协方差", xfc)
        # 计算特征值和特征向量
        tzz, tzxl = np.linalg.eig(xfc)
        print("特征值", tzz)
        print("特征向量", tzxl)
        # 对特征值从大到小排序，结果是索引值按特征值的排序
        index_sort = np.argsort(-tzz)
        # 取前k个特征向量
        tzxl = tzxl[:, index_sort[: self.k]]
        print("取k后特征向量", tzxl)
        # 原数据*特征向量=降维后的数据
        result = np.dot(self.X, tzxl)

        # 均值处理，否则处理结果和sklearn计算的值会有差别
        result-=np.mean(result, axis=0)

        return result


def pca_test():
    X = np.array(
        [
            [10, 15, 29],
            [15, 46, 13],
            [23, 21, 30],
            [11, 9, 35],
            [42, 45, 11],
            [9, 48, 5],
            [11, 21, 14],
            [8, 5, 15],
            [11, 12, 21],
            [21, 20, 25],
        ]
    )

    pca = PCA(X, 2)
    result = pca.to_PCA()
    print(result)


pca_test()
