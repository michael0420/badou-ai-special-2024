import numpy as np
import cv2 as cv

def WarpPerspectiveMatrix(src, dst):
    nums = src.shape[0]
    A = np.zeros((2*nums, 8))
    B = np.zeros((2 * nums, 8))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i] = B_i[0]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)#创建矩阵
    #A的逆乘以B得到变换矩阵
    warpMatrix = A.I * B

    warpMatrix = np.array(warpMatrix).T[0]
    #插入a_33 = 1
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

#
img = cv.imread('photo.jpg')
newImg = img.copy()
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
#api
m = cv.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
#手动实现
m = WarpPerspectiveMatrix(src, dst)
print(m)
result = cv.warpPerspective(newImg, m, (337, 408))
cv.imshow("src", img)
cv.imshow("dst", result)
cv.waitKey(0)