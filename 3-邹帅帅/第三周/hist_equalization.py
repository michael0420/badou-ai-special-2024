#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

# 获取灰度图像
img = cv2.imread('E:\image\lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
# plt.subplot(121)
# plt.hist(img_hist.ravel(), 256)
#
# plt.subplot(122)
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow('Histogram Equalization', np.hstack([gray, dst]))


# 彩色图像直方图均衡化

(b, g, r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

result = cv2.merge((bh, gh, rh))
cv2.imshow('rgb', result)
cv2.waitKey()
