import numpy as np
import random as rd
import cv2
import math
import matplotlib.pyplot as plt
###实现canny detail
def CannyDetail(img, kernel_size, sigma):

   gauss_filter = GaussFilter(kernel_size, sigma)
   conv_img = Convolution(img, gauss_filter)
   res_img = jizhiyizhi(conv_img)
   return res_img


def GaussFilter(kernel_size = 5, sigma = 0.5):
    max_idx = kernel_size // 2
    idx = np.linspace(-max_idx, max_idx, kernel_size)
    Y, X = np.meshgrid(idx, idx)
    print(X ** 2)
    gauss_filter = (1/(2*math.pi*sigma**2)) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    gauss_filter /= np.sum(gauss_filter)
    return gauss_filter

def Convolution(img, gauss_filter):
    f = len(gauss_filter)
    h, w = img.shape[0:2]
    out_img = np.zeros([h, w], img.dtype)

    ###PADING
    p = int((f - 1) / 2)
    padingImg = np.zeros([h + p * 2, w + p * 2], img.dtype)

    for i in range(h):
        for j in range(w):
            padingImg[i + p][j + p] = img[i][j]

    for i in range(h):
        for j in range(w):
            temp1 = padingImg[i:i+f, j:j+f]
            tempx = np.sum(temp1 * gauss_filter)
            out_img[i][j] = tempx
    return out_img

def jizhiyizhi(img):

    dx, dy = img.shape

    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_tidu_x = np.zeros([dx, dy])
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros([dx, dy])
    img_pad = np.pad(img, ((1, 1), (1, 1)), 'constant')

    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x

    zhan = []
    jizhi_img = np.zeros([dx, dy], img_tidu.dtype)

    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]
            if angle[i, j] <= -1:
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                jizhi_img[i, j] = img_tidu[i, j]

    low_b = img_tidu.mean() * 0.5
    high_b = low_b * 3

    for i in range(1, jizhi_img.shape[0]-1):
        for j in range(1, jizhi_img.shape[1]-1):
            if jizhi_img[i, j] >= high_b:
                jizhi_img[i, j] = 255
                zhan.append([i, j])
            elif jizhi_img[i, j] <= low_b:
                jizhi_img[i, j] = 0

    while len(zhan) > 0:
        tempI, tempJ = zhan.pop()
        temp_img = jizhi_img[tempI - 1:tempI + 2, tempJ - 1:tempJ + 2]
        for i in range(len(temp_img)):
            for j in range(len(temp_img[0])):
                if temp_img[i][j] > low_b and temp_img[i][j] < high_b:
                    jizhi_img[tempI + i - 1][tempJ + j - 1] = 255
                    zhan.append([tempI + i - 1, tempJ + j - 1])

    for i in range(jizhi_img.shape[0]):
        for j in range(jizhi_img.shape[1]):
            if jizhi_img[i, j] != 0 and jizhi_img[i, j] != 255:
                jizhi_img[i, j] = 0

    return jizhi_img



gray_img = cv2.imread('lenna.png', 0)
# RGB_img = cv2.imread('lenna.png',cv2.COLOR_BGR2RGB)
canny_img = CannyDetail(gray_img, 5, 0.5)
cv2.imshow("canny_img", canny_img)
# # cv2.imshow("bilImg", bil_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# a = np.array([257.22])
# print(a.astype(np.uint8))

###实现透视变换
def Change(src, dst, img, size):
    m = WarpPerspectiveMatrix(src, dst)
    c_img = cv2.warpPerspective(img, m, size)

    return c_img


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))
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

    warpMatrix = A.I * B

    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


def pective_transformation(img, warpMatrix, size):
    height, weight, channel = img.shape

    new_img = np.zeros((size[1], size[0], channel))
    warpMatrix = np.mat(warpMatrix)
    warpMatrix = warpMatrix.reshape((3, 3))

    for i in range(size[0]):
        for j in range(size[1]):

            goal = [i, j, 1]
            goal = np.mat(goal)
            goal = goal.reshape((3, 1))

            img_point = warpMatrix.I * goal
            img_point = img_point.tolist()

            x = int(np.round(img_point[0][0] / img_point[2][0]))
            y = int(np.round(img_point[1][0] / img_point[2][0]))

            if y >= img.shape[0]:
                y = img.shape[0] - 1
            if x >= img.shape[1]:
                x = img.shape[1] - 1
            if y < 0:
                y = 0
            if x < 0:
                x = 0
            new_img[j, i] = img[y, x]
    new_img = new_img.astype(np.uint8)
    return new_img

img = cv2.imread('photo1.jpg')

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

c_img = Change(src, dst, img, [337, 488])

cv2.imshow("src", img)
cv2.imshow("result", c_img)
cv2.waitKey(0)

