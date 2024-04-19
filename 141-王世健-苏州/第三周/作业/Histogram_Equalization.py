#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png", 1)

# 灰色图像直方图均衡化
def histogram_equalization(img):
    # 读取图像


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #转换为灰度图像
    # cv2.imshow("image_gray", gray)

    # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(gray) #直方图均衡化

    # 直方图
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

    return dst, gray



# 彩色直方图均衡化
def Chroma_histogram_equalization(img):

    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))

    return result



dst, gray  = histogram_equalization(img)
plt.subplot(1, 2, 1)
plt.hist(gray.ravel(), 256)
plt.subplot(1, 2, 2)
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
result = Chroma_histogram_equalization(img)
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)


