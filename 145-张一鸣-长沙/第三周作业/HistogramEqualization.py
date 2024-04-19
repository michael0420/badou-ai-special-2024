# coding = utf-8

'''
        直方图均衡化
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

'''
# 彩色图像
img = cv2.imread('lenna.png')
plt.subplot(221)
plt.imshow(img)

# 彩色图像分三通道均衡化、生成直方图
(B, G, R) = cv2.split(img)
Bhist = cv2.equalizeHist(B)
Ghist = cv2.equalizeHist(G)
Rhist = cv2.equalizeHist(R)

hist_1 = cv2.calcHist([Bhist], [0], None, [256], [0, 256])
hist_2 = cv2.calcHist([Ghist], [0], None, [256], [0, 256])
hist_3 = cv2.calcHist([Rhist], [0], None, [256], [0, 256])

plt.figure()
plt.hist(hist_1, 256)
plt.show()

# 组合三通道均衡化结果
res = cv2.merge((Bhist, Ghist, Rhist))
cv2.imshow('彩图均衡化', res)
cv2.waitKey(0)
'''

# 灰度化+均衡化
# 调用接口灰度化
src = cv2.imread('lenna.png')
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey(0)

# 均衡化
hist = cv2.equalizeHist(gray)
cv2.imshow('equl_hist', hist)
cv2.waitKey(0)

# 调用接口生成直方图
h_result = cv2.calcHist([hist], [0], None, [256], [0, 256])
plt.figure()
plt.hist(hist.ravel(), 256)
plt.show()
