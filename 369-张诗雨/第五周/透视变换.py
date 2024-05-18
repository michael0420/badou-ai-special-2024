import cv2
import numpy as np


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    img = cv2.imread('photo1.jpg')
    img_copy = img.copy()
    '''
    注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
    '''
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

    warpMatrix = cv2.getPerspectiveTransform(src, dst)
    print("warpMatrix:")
    print(warpMatrix)
    result = cv2.warpPerspective(img_copy, warpMatrix, (337, 488))

    warpMatrix2 = WarpPerspectiveMatrix(src, dst)
    result2 = cv2.warpPerspective(img_copy, warpMatrix2, (337, 488))

    cv2.imshow("src", img)
    cv2.imshow("result1", result)
    cv2.imshow("result2", result2)
    cv2.waitKey(0)