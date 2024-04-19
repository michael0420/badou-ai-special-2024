'''
作业2：
实现直方图均衡化
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读图，并灰度化
img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
img_hist = cv2.equalizeHist(img_gray)

# 直方图
hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

plt.figure()
plt.hist(img_hist.ravel(), 256)
plt.show()


# 彩色图像直方图均衡化
# 通道分离，并在每个通道上实现均衡
b, g, r = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

# 合并通道
img_hist2 = cv2.merge((bH, gH, rH))


# plt.subplot(221)
# plt.imshow(img, )
# plt.subplot(222)
# plt.imshow(img_gray, cmap='gray')
# plt.subplot(223)
# plt.imshow(img_hist, cmap='gray')
# plt.subplot(224)
# plt.imshow(img_hist2)
# plt.show()

cv2.imshow('img', img_hist2)
cv2.waitKey(0)

