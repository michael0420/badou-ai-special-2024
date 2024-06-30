import numpy as np


class PCA(object):

    def __init__(self, X, k):
        # 原始矩阵
        self.src = X
        # 目标维度数
        self.k = k
        # 样本个数
        self.m = X.shape[0]

        # 计算
        # 中心化矩阵
        self._centralized()
        # 协方差矩阵
        self._cov_()
        # 特征值
        self._feature_matrix_()
        # 算出降维Z
        self._cal_dst_()

    def _centralized(self):
        """中心化"""
        # 计算每个特征的平均值
        mean = np.mean(self.src, axis=0)
        # self.center = [np.mean(a) for a in self.src]
        print("特征均值: \n", mean)
        self.center = self.src - mean
        print("中心化矩阵: \n", self.center)

    def _cov_(self):
        """计算协方差"""
        self.cov = np.dot(self.center.T, self.center) / (self.m - 1)
        print("协方差矩阵: \n", self.cov)

    def _feature_matrix_(self):
        """计算特征值和特征向量, 算出降维转换矩阵"""
        self.feature_val, self.feature_vec = np.linalg.eig(self.cov)
        print("特征值:\n", self.feature_val)
        print("特征向量:\n", self.feature_vec)
        # 对特征值降序排序
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1 * self.feature_val)
        print("索引序列 \n", ind)
        # 取出对应的特征向量
        UT = [self.feature_vec[:, ind[i]] for i in range(self.k)]
        # UT = [self.feature_vec[:, ind[i]:(ind[i] + 1)] for i in range(self.k)]
        self.feature_k = np.transpose(UT)

        print("%d阶降维转换矩阵: \n" % self.k, self.feature_k)

    def _cal_dst_(self):
        self.dst = np.dot(self.src, self.feature_k)
        print("降维的结果: \n", self.dst)
        return self.dst



if __name__ == '__main__':
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
    pca = PCA(X, 2)

