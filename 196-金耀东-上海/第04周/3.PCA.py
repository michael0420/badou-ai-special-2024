import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._base import load_iris

class PCA:

    def __init__(self):
        # 原始数据（行向量）组成的矩阵
        self.src_data = None
        # 中心化后数据后的矩阵
        self.centr_data = None
        # 特征的协方差矩阵
        self.cov_data = None
        # 所有特征（列向量）组成的矩阵
        self.all_components = None

    def _calc_center(self):
        self.centr_data = self.src_data - np.mean(self.src_data, axis=0)

    def _calc_covarirance(self):
        self.cov_data = np.dot( self.centr_data.T, self.centr_data) / (self.centr_data.shape[0] - 1)

    def _calc_all_components(self):
        feature_values , feature_vectors = np.linalg.eig( self.cov_data )
        indicies = np.argsort(-feature_values)
        # 将分解得到的特征向量按照特征值降序排列
        self.all_components = feature_vectors[:,indicies]

    def analyze(self, data):
        # 原始数据
        self.src_data = data
        # 计算中心化矩阵
        self._calc_center()
        # 计算协方差矩阵
        self._calc_covarirance()
        # 计算主成分
        self._calc_all_components()

    def transform_data(self, n_dim):
        # 提取分特征值最大的前n_dim维作为转换矩阵
        transform_matrix = self.all_components[:,:n_dim]
        # 返回数据转化结果
        return np.dot( self.src_data, transform_matrix )

def display_data(data, labels):
    red_x, red_y = [], []
    green_x, green_y = [], []
    blue_x, blue_y = [], []
    for i in range( len(labels) ):
        if labels[i] == 0:
            red_x.append(data[i][0])
            red_y.append(data[i][1])
        elif labels[i] == 1:
            green_x.append(data[i][0])
            green_y.append(data[i][1])
        else:
            blue_x.append(data[i][0])
            blue_y.append(data[i][1])

    plt.scatter(red_x, red_y, c='r')
    plt.scatter(green_x, green_y, c='g')
    plt.scatter(blue_x, blue_y, c='b')
    plt.show()

if __name__ == "__main__":
    # 加载鸢尾花测试数据
    src_data , labels = load_iris(return_X_y=True)
    # 转换后的数据维度
    n_dim = 2

    # 加载PCA算法
    my_pca = PCA()
    # 对数据进行PCA主成分分析
    my_pca.analyze(src_data)
    # 将数据降到n_dim维
    dst_data = my_pca.transform_data(n_dim)

    # 查看降维效果
    display_data(dst_data, labels)

