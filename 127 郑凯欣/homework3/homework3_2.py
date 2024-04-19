# -*- coding: utf-8 -*-
'''
@ Author: Hayne
@ Time: 2024/04/04
@ Name: Histogram Equalization
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

# 灰度图cv2.cvtColor()
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.waitKey()

# 直方图均衡化cv2.equalizeHist()
out = cv2.equalizeHist(gray)

# 绘制直方图：
# 1、matplotlib.pyplot中的hist(数组，直方图列数)函数能够直接绘制直方图，ravel()将二维数组降维一维
plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()

# 2、OpenCV 中的 cv2.calcHist()函数能够计算统计直方图，然后在此基础上绘制图像的直方图。
# 计算直方图cv2.calcHist()
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist, color='b')
plt.show()

# 画图
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(gray, cmap='gray')

plt.subplot(223)
plt.imshow(out, cmap='gray')

plt.show()

# 彩色直方图均衡化
img = cv2.imread("lenna.png")

# 将图像分解为三通道cv2.split(img)，分别做均衡化
(b, g, r) = cv2.split(img)
b_hist = cv2.equalizeHist(b)
g_hist = cv2.equalizeHist(g)
r_hist = cv2.equalizeHist(r)
# 合并通道cv2.merge(b,g,r)
result = cv2.merge((b_hist, g_hist, r_hist))
# plt.imshow的三通道顺序与cv2不一致
result = cv2.merge((r_hist, g_hist, b_hist))

plt.subplot(121)
img = plt.imread("lenna.png")
plt.imshow(img)
plt.subplot(122)
plt.imshow(result)
plt.show()
# cv2.imshow("dst_rgb", result)
# cv2.waitKey(0)
