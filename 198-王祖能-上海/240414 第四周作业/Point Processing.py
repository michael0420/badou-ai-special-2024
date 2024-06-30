'''
图像增强：点处理，领域处理。
点处理：线性linear变化，指数logarithm变化，幂power变化
'''
import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt

# 进行线性增强点处理


def linear_process(src, a, b):
    '''
    :param src: 读取图像矩阵
    :param a: 权重
    :param b: 偏置
    :return:
    '''
    h, w, c = src.shape
    dst = src.copy()
    for z in range(c):
        for i in range(h):
            for j in range(w):
                temp = a * src[i, j, z] + b
                if temp < 0:
                    temp = 0
                elif temp > 255:
                    temp = 255
                dst[i, j, z] = temp
    return dst


img = cv2.imread('lenna.png')
img_linear_process = linear_process(img, 2, 0)
cv2.imshow('img_linear_process', np.hstack([img, img_linear_process]))
cv2.waitKey()
cv2.destroyAllWindows()

# 进行对数增强点处理


def logarithm_process(src, a):
    '''
    :param src: 读取图像矩阵
    :param a: 权重
    :return:
    '''
    h, w = src.shape
    dst = src.copy()
    for i in range(h):
        for j in range(w):
            temp = a * np.log(dst[i, j] + 1)
            if temp < 0:
                temp = 0
            elif temp > 255:
                temp = 255
            dst[i, j] = temp
    return dst


img = cv2.imread('lenna.png', 0)
img_logarithm_process = logarithm_process(img, 20)  # 随着a的放大，集中的像素范围向右平移
cv2.imshow('img_logarithm_process', np.hstack([img, img_logarithm_process]))
cv2.waitKey()
cv2.destroyAllWindows()

HIST1 = cv2.calcHist([img], [0], None, [256], [0, 256])
HIST2 = cv2.calcHist([img_logarithm_process], [0], None, [256], [0, 256])
plt.plot(HIST1, color='red')
plt.plot(HIST2, color='blue')  # 可以利用对数图像加强，将高灰度向低灰度集中log100才=2,图片极黑
plt.show()

# 进行指数增强点处理


def power(src, a, gama):
    '''

    :param src:输入源图
    :param gama: 幂指数
    :return:
    '''
    dst = src.copy()
    h, w = dst.shape
    for i in range(h):
        for j in range(w):
            temp = a * np.power(dst[i, j], gama)
            if temp < 0:
                temp = 0
            elif temp > 255:
                temp = 255
            dst[i, j] = temp
    return dst


img = cv2.imread('lenna.png', 0)
img_power = power(img, 1, 1.05)  # ＞1整体变亮，向右偏移，灰度越高偏移越明显
cv2.imshow('img_power', np.hstack([img, img_power]))
cv2.waitKey()

HIST1 = cv2.calcHist([img], [0], None, [256], [0, 256])
HIST2 = cv2.calcHist([img_power], [0], None, [256], [0, 256])
plt.plot(HIST1, color='red')
plt.plot(HIST2, color='blue')  # 可以利用对数图像加强，将高灰度向低灰度集中log100才=2,图片极黑
plt.show()
