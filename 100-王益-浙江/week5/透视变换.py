import numpy as np
import cv2


def WarpPerspectiveMatrix(src, dst):
    A = np.zeros((8, 8))
    B = np.zeros((8, 1))

    for i in range(4):
        x, y = src[i]
        x_p, y_p = dst[i]
        A[i * 2] = [x, y, 1, 0, 0, 0, -x_p * x, -x_p * y]
        A[i * 2 + 1] = [0, 0, 0, x, y, 1, -y_p * x, -y_p * y]
        B[i * 2] = x_p
        B[i * 2 + 1] = y_p

    A_inv = np.linalg.inv(A)
    M = A_inv.dot(B)
    M = np.insert(M, M.shape[0], 1, axis=0)
    # 这句代码的意思是
    M = M.reshape(3,3)
    return M


img = cv2.imread('img/photo1.jpg')
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
Matrix = WarpPerspectiveMatrix(src, dst)
img_warp = cv2.warpPerspective(img, Matrix, (337, 488))
cv2.imshow('original', img)
cv2.imshow('warped', img_warp)
cv2.waitKey(0)
