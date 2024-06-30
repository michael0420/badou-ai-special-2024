import numpy as np


def WarpPerspectiveMatrix(src, dst):
    print('warpMatrix')
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    x_nums = src.shape[0]
    A = np.zeros((2 * x_nums, 8))
    B = np.zeros((2 * x_nums, 1))
    for i in range(0, x_nums):
        A_i = src[i, :]
        B_i = dst[i, :]

        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,-A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))

    return warpMatrix


if __name__ == '__main__':

    src_point = [[12.0, 457.0], [50.0, 500.0], [800.0, 291.0], [1200.0, 300.0]]
    dst_point = [[50.0, 700.0], [20.0, 300.0], [500.0, 48.0], [660.0, 60.0]]
    src = np.array(src_point)
    dst = np.array(dst_point)
    warpMatrix = WarpPerspectiveMatrix(src, dst)

    print(warpMatrix)
