#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('image-gray', gray)

# 灰度图像直方图均衡化
new1 = cv2.equalizeHist(gray)

# 直方图

hist = cv2.calcHist([new1], [0], None, [256], [0, 256])
plt.figure()
plt.hist(new1.ravel(), 256)
plt.show()

cv2.imshow('Histogram', np.hstack([gray, new1]))
cv2.waitKey(0)

# 彩色图像直方图均衡化
img = cv2.imread('lenna.png', 1)
cv2.imshow('src', img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)

# 合并每一个通道
result = cv2.merge((bh, gh, rh))
cv2.imshow('dst_rgb', result)
cv2.waitKey(0)

