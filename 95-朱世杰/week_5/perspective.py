"""
实现透视变换
@Author： zsj
"""

import numpy as np
import cv2
src = np.array([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.array([[0, 0], [337, 0], [0, 488], [337, 488]])

nums = src.shape[0]
A = np.zeros((2 * nums, 8))
B = np.zeros((2 * nums, 1))
for i in range(0, nums):
    src_i = src[i, :]
    dst_i = dst[i, :]
    A[2 * i, :] = [src_i[0], src_i[1], 1, 0, 0, 0,
                   -src_i[0] * dst_i[0], -src_i[1] * dst_i[0]]
    B[2 * i] = dst_i[0]

    A[2 * i + 1, :] = [0, 0, 0, src_i[0], src_i[1], 1,
                       -src_i[0] * dst_i[1], -src_i[1] * dst_i[1]]
    B[2 * i + 1] = dst_i[1]

A = np.mat(A)
# 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
# 把1插入到最后一个，并重整为3*3的矩阵
warpMatrix = np.append(warpMatrix.A1, 1)
# 算出透视变换矩阵
warpMatrix = warpMatrix.reshape(3, 3)


img = cv2.imread('photo1.jpg')
result = cv2.warpPerspective(img, warpMatrix, [337, 488])
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)