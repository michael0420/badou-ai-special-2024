# -*- coding: utf-8 -*-
"""
@author: baotongxing

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化
# 也可使用系统库函数rgb2gray(img)) 实现灰度化
def imageGray(img):
    h, w = img.shape[:2];
    img_gray = np.zeros([h, w], img.dtype);
    for i in range(h):
        for j in range(w):
            m = img[i, j];
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3);   # 顺序BGR
    return img_gray

# 打印灰度化
def printGraycv2():
    # cv2 加载图像
    img = cv2.imread("pic/lenna.png")
    # 调用置灰函数
    img_gray = imageGray(img)

    print(img_gray)
    print("image show gray: %s" % img_gray)
    cv2.imshow("image show gray", img_gray)

# 打印plt灰度化
def printGrayplt(img):
    # 灰度化
    img_gray = rgb2gray(img)
    plt.subplot(222)
    plt.imshow(img_gray, cmap='gray')
    print("---image gray----")
    print(img_gray)
    return img_gray

# 打印plt原图展示
def printimageplt(img):
    plt.subplot(221)
    # img = cv2.imread("lenna.png", False)
    plt.imshow(img)
    print("---image lenna----")
    print(img)

# 打印plt二值化
def printBinplt(img):
    img_binary = np.where(img >= 0.5, 1, 0)
    print("-----imge_binary------")
    print(img_binary)
    print(img_binary.shape)

    plt.subplot(223)
    plt.imshow(img_binary, cmap='gray')
    plt.show()

# 图片展示
def display():
    # 灰度化
    printGraycv2();
    # 在画板上加载图片
    img = plt.imread("pic/lenna.png")
    printimageplt(img)
    # 在画板上灰度化
    img_gray = printGrayplt(img)
    # 在画板上二值化
    printBinplt(img_gray)


# 图像展示
display();