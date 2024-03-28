'''
作业1：
图像灰度化和二值化
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray


def showimg(img):
    cv2.imshow('show img', img)
    cv2.waitKey(0)
    cv2.destroyWindow('show img')


# 使用原始计算的方式，实现图像的灰度化
img1 = cv2.imread('lenna.png')              # 读入图片
h, w = img1.shape[:2]                       # 获取图片的矩阵大小
img1_gray = np.zeros([h, w], img1.dtype)    # 定义灰度图的矩阵大小
for i in range(h):
    for j in range(w):
        ppi_bgr = img1[i, j]                # 获取img1在坐标[i, j]的bgr像素值
        img1_gray[i, j] = int(              # 图像灰度化计算
            ppi_bgr[0]*0.114 + ppi_bgr[1]*0.587 + ppi_bgr[2]*0.299)

showimg(img1_gray)

# 以灰度图的方式读入图片
img2_gray = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
showimg(img2_gray)

# 学习使用二值化
img1_binary = img1_gray / 225               # 归一化
for i in range(h):                          # 遍历矩阵
    for j in range(w):
        if img1_binary[i, j] <= 0.5:        # 进行二值化判断
            img1_binary[i, j] = 0
        else:
            img1_binary[i, j] = 1
showimg(img1_binary)

# 使用numpy进行判断
plt.subplot(221)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # 将opencv读入的bgr图片转为rgb
plt.imshow(img1)
img1_gray2 = rgb2gray(img1)                 # 使用rgb2gray功能，转得的矩阵会归一化。
plt.subplot(222)
plt.imshow(img1_gray2, cmap='gray')
img1_binary2 = np.where(img1_gray2 < 0.5, 0, 1)
plt.subplot(223)
plt.imshow(img1_binary2, cmap='gray')
plt.show()


