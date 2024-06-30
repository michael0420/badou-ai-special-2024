import numpy as np
import sklearn.decomposition as dp

if __name__ == '__main__':
    matrix_1 = np.array([[10, 15, 29],
                         [15, 46, 13],
                         [23, 21, 30],
                         [11, 9, 35],
                         [42, 45, 11],
                         [9, 48, 5],
                         [11, 21, 14],
                         [8, 5, 15],
                         [11, 12, 21],
                         [21, 20, 25]])
    matrix_2 = dp.PCA(n_components=2).fit_transform(matrix_1)

    print(matrix_2)
    # pca = dp.PCA(n_components=2)
    # pca.fit(matrix_1)
    # matrix_2 = pca.fit_transform(matrix_1)
    # print(matrix_2)
