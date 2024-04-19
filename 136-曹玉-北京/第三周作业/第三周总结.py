#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('lenna.png',1)
# 双线性插值
resized_image = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
resized_image2 = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)

# 灰度图像直方图均衡化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst1 = cv2.equalizeHist(gray)

# 直方图数据处理和打印
hist = cv2.calcHist([dst1],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst1.ravel(), 256)
plt.show()

# 彩色图像直方图均衡化
(b, g, r) = cv2.split(resized_image)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))

# Sobel边缘检测
img2 = cv2.imread("lenna.png", 0)
x = cv2.Sobel(img2, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img2, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 正常显示图片
cv2.imshow('lenna',img)
# cv2.imshow('bilinear interp800',resized_image)
cv2.imshow('bilinear interp700',resized_image2)
# cv2.imshow("dst_rgb", result)
# cv2.imshow("absX", absX)
# cv2.imshow("absY", absY)
# cv2.imshow("Result", dst)
# cv2.imshow("old", gray)
# cv2.imshow("new", dst1)

# 水平堆叠图片
cv2.imshow("Sobel test",np.hstack([absX,absY,dst])) 
cv2.imshow("equalizeHist",np.hstack([resized_image,result])) 
cv2.imshow("gray equalizeHist",np.hstack([gray,dst1]))
cv2.waitKey()

