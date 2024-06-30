import numpy as np


class PCA(object):
    """
    主成分分析
    """

    def __init__(self, X, k, is_print=True):
        """
        :param X: 输入数据集
        :param k: 降维后的维度
        """

        self.X = X
        self.k = k

        self.is_print = is_print

        # 中心化
        self.CenterX = []
        # 协方差
        self.XFC = []
        # 求特征向量
        self.TZXL = []
        # 原数据*特征向量，求出PCA结果
        self.result = []

    def _to_CenterX(self):
        """
        中心化：1）求平均值，2）所有数据-平均值
        :return:
        """
        self.print_info(f"原始数据：{self.X}")

        mean = [np.mean(a) for a in self.X.T]

        self.CenterX = self.X - mean

        self.print_info(f"中心化数据：{self.CenterX}")

    def _to_XFC(self):
        """
        协方差：中心化矩阵转置*中心化矩阵，除以数据集个数
        :return:
        """
        # 样本总数
        n = self.X.shape[0]
        self.XFC = np.dot(self.CenterX.T, self.CenterX) / (n - 1)

        self.print_info(f"协方差：{self.XFC}")

    def _to_TZXL(self):
        """
        求特征值、特征向量
        :return:
        """
        tzz, tzxl = np.linalg.eig(self.XFC)
        self.print_info(f"特征值：{tzz}")
        self.print_info(f"特征向量：{tzxl}")

        # 对特征值索引排序，结果是按值倒序后的索引
        sort_index = np.argsort(-1 * tzz)
        # 取前k个最大的特征向量，注意这个时候的矩阵是反过来的
        self.TZXL = [tzxl[:, sort_index[i]] for i in range(self.k)]
        # 转置回来
        self.TZXL = np.transpose(self.TZXL)

        self.print_info(f"取k后特征向量：{self.TZXL}")

    def to_PCA(self):
        """
        PCA计算最终结果
        """

        self._to_CenterX()
        self._to_XFC()

        # self.XFC = np.array(
        #     [
        #         [3, 2],
        #         [1, 4],
        #     ]
        # )

        self._to_TZXL()

        self.result = np.dot(self.X, self.TZXL)

        # 均值处理，否则处理结果和sklearn计算的值会有差别
        self.result -= np.mean(self.result, axis=0)

        self.print_info(f"PCA结果：{self.result}")
        return self.result

    def print_info(self, info):
        if self.is_print:
            print(info)


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

    k = 2
    pca = PCA(X, k)
    pca.to_PCA()


pca_test()
