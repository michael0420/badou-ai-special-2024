# _*_ coding: UTF-8 _*_
# @Time: 2024/4/24 17:06
# @Author: iris
# @Email: liuhw0225@126.com
import numpy as np


def warpMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = dst.shape[0]
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))
    for i in range(nums):
        A_I = src[i, :]
        B_I = dst[i, :]
        x0 = A_I[0]
        y0 = A_I[1]
        X0 = B_I[0]
        Y0 = B_I[1]

        A[2 * i, :] = [x0, y0, 1, 0, 0, 0, -x0 * X0, -y0 * X0]
        B[2 * i] = B_I[0]

        A[2 * i + 1, :] = [0, 0, 0, x0, y0, 1, -x0 * Y0, -y0 * Y0]
        B[2 * i + 1] = B_I[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    matrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    matrix = np.array(matrix).T[0]
    matrix = np.insert(matrix, matrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    matrix = matrix.reshape((3, 3))
    return matrix


if __name__ == '__main__':
    """
        warpMatrix 根据已知八组点数据获得变换矩阵
    """
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])
    data = warpMatrix(src, dst)
    print(data)
