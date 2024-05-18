import numpy as np


def warpMatrix(source, target):
    if ((source.shape[0] == target.shape[0] and source.shape[0] >= 4) is False):
        return None
    coordinates = source.shape[0]

    A = np.zeros((coordinates * 2, coordinates * 2))
    B = np.zeros((coordinates * 2, 1))

    # 求A B矩阵
    for i in range(coordinates):
        A[i * 2, :] = [source[i, :][0], source[i, :][1], 1, 0, 0, 0, -source[i, :][0] * target[i, :][0],
                       -source[i, :][1] * target[i, :][0]]
        B[i * 2] = target[i, :][0]

        A[i * 2 + 1, :] = [0, 0, 0, source[i, :][0], source[i, :][1], 1, -source[i, :][0] * target[i, :][1],
                           -source[i, :][1] * target[i, :][1]]
        B[i * 2 + 1] = target[i, :][1]

    A = np.mat(A)  # 将数据转成矩阵

    # A * warpMatrix=B ->  warpMatrix=A.I*B

    warp_matrix = A.I * B

    warp_matrix = np.append(warp_matrix, [[1.]], axis=0)

    return warp_matrix.reshape((3, 3))


if __name__ == '__main__':
    source_coordinates = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    target_coordinates = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])

    matrix_1 = warpMatrix(source_coordinates, target_coordinates)

    print(matrix_1)
