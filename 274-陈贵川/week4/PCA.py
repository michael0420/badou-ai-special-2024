import numpy as np


class PCATest:

    def __init__(self, target_dimension):
        self.target_dimension = target_dimension

    def trans_dimension(self, src_data):
        print("PCA_TEST")
        if src_data.shape[1] <= self.target_dimension:
            print("ERROR：The target dimension should be smaller than the input dimension ")
            return src_data
        # 求均值
        # 中心化
        # 求协方差矩阵
        # 求协方差矩阵的特征之及特征向量
        # 按特征值排序取K个合成特征向量的降维空间
        # 映射到新的特征空间
        data_mean = src_data.mean(axis=0)
        print(data_mean)
        data_centralized = src_data - data_mean
        data_cov = np.dot(data_centralized.T, data_centralized) / data_centralized.shape[0]
        eig_values, eig_vectors = np.linalg.eig(data_cov)
        eig_sorted_idx = np.argsort(-eig_values)
        data_new = eig_vectors[:, eig_sorted_idx[:self.target_dimension]]
        data_result = np.dot(src_data, data_new)
        return data_result


if __name__ == "__main__":
    # test
    pca = PCATest(target_dimension=4)
    test_mat = np.array(
        [[-1, 2, 66, -1],
         [-2, 6, 58, -1],
         [-3, 8, 45, -2],
         [1, 9, 36, 1],
         [2, 10, 62, 1],
         [3, 5, 83, 2]])  # 维度为4
    mat_new = pca.trans_dimension(test_mat)
    print(mat_new)
