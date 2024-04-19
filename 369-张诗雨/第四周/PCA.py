import numpy as np


def PCA(X, k_components):
    # 去中心化
    B = X - X.mean(axis=0)
    # 求协方差矩阵
    covariance = np.dot(B.T, B) / B.shape[0]
    # 求特征值，特征向量
    eig_vals, eig_vectors = np.linalg.eig(covariance)
    # 获得降序排列特征值的序号
    idx = np.argsort(-eig_vals)
    # 降维特征矩阵
    P = eig_vectors[:, idx[:k_components]]
    # 对B进行降维
    return np.dot(B, P)


if __name__ == '__main__':
    X = np.array(
        [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
    newX = PCA(X, 2)
    print(newX)
