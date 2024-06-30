import numpy as np


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4  # 最少要有4个坐标，而且要一一对应

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        print(A_i)
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    A = np.mat(A)  # 转为矩阵
    warpMatrix = A.I * B  # 求解

    warpMatrix = np.array(warpMatrix).T[0]  # 转置
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入值
    warpMatrix = warpMatrix.reshape((3, 3))  # 重塑成3*3的矩阵
    return warpMatrix


if __name__ == '__main__':
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    # src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    # dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
