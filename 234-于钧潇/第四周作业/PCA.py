import numpy as np
import sklearn.decomposition as dp

def PCA(original_simple, K):
    # 均值化
    central_simple = original_simple - original_simple.mean(axis=0)
    # 样本数
    m = original_simple.shape[0]
    # 求协方差矩阵
    COV = np.dot(central_simple.T, central_simple) / m
    # 求特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(COV)
    # 排序，取前K个 组成特征矩阵W
    idx = np.argsort(-eigenvalues)
    W = eigenvectors[:, idx[:K]]
    # 返回均值化矩阵乘特征矩阵
    return np.dot(central_simple, W)

if __name__ == '__main__':
    # 原始样本矩阵
    original_simple = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])

    print("原始数据：", original_simple)
    XW = PCA(original_simple, 2)
    print("降维后的数据：", XW)