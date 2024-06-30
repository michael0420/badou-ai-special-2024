'''
根据手写矩阵 求透视变换矩阵
@zeno wang
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt


def warpMatrix(src, dst):
    assert src.shape == dst.shape and src.shape[0] >= 4  # 判断只有满足，才继续向下执行

    num = src.shape[0]
    A = np.zeros([num * 2, 8])
    B = np.zeros([num * 2, 1])  # A * warpMatrix = B
    for i in range(0, num):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i] = B_i[0]
        B[2 * i + 1] = B_i[1]
    # print(type(A), A.shape)  # 数组类型
    A = np.mat(A)  # 数组类型可以转置，但不能求逆矩阵，需要强制转化为矩阵后运算
    # print(type(A), A.shape)  # 矩阵类型
    warp = A.I * B  # 得到八个参数值，a33 = 1，再整理成3*3矩阵形式
    # print(warp.T, warp.T.shape)
    warp = np.array(warp).T

    # print(warp, warp.shape)
    # warp = np.insert(warp, warp.shape[1], values=1.0, axis=1)  # 老师的方法，按位置插入a33=1.0

    warp = np.array(warp)
    warp = np.append(warp, 1.0)  # 列矩阵重新转回数组类型以后在最后追加a33=1，其他位置可以用insert插入，再改变形状
    warp = warp.reshape([3, 3])

    return warp


if __name__ == '__main__':
    src = np.array([[10.0, 457.0],
                    [395.0, 291.0],
                    [624.0, 291.0],
                    [1000.0, 457.0]])
    dst = np.array([[46.0, 920.0],
                    [46.0, 100.0],
                    [600.0, 100.0],
                    [600.0, 920.0]])
    warp = warpMatrix(src, dst)
    print('warpMatrix:\n', warp)
