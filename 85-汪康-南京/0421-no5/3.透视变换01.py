# -*- coding: utf-8 -*-
'''@Time: 2024/5/11 21:09

'''
import numpy as np
# 1.手写warpMatrix
def WarpPerspectiveMatrix(src,dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0,nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]
    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    # print(f'warpMatrix:{warpMatrix}')
    # 之后为结果的后处理
    x = np.array(warpMatrix)
    # print(f'x:{x}')
    warpMatrix = np.array(warpMatrix).T[0]
    print(f'warpMatrix:{warpMatrix}')
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    print('warpMatrix')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)

    import cv2
    # 2.使用cv2自带api
    img = cv2.imread('photo1.jpg')

    result3 = img.copy()

    '''
    注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
    '''
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    print(img.shape)
    # 生成透视变换矩阵；进行透视变换
    m = cv2.getPerspectiveTransform(src, dst)
    print("warpMatrix:")
    print(m)
    result = cv2.warpPerspective(result3, m, (337, 488))
    cv2.imshow("src", img)
    cv2.imshow("result", result)
    cv2.waitKey(0)

