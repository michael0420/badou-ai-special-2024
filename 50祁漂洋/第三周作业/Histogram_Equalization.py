"""
@author: QiPiaoYang

直方图均衡化
函数原型：equalizeHist(src, dst=None)

"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# 灰度图像直方图均衡化
grayImg = cv2.imread("../lenna.png", 0)
cv2.imshow("image_gray", grayImg)
cv2.waitKey(0)

dst = cv2.equalizeHist(grayImg)

hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([grayImg, dst]))
cv2.waitKey(0)

# 彩色图像直方图均衡化
colorImg = cv2.imread('../lenna.png', 1)
cv2.imshow('color_image', colorImg)
cv2.waitKey(0)

(b, g, r) = cv2.split(colorImg)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH, gH, rH))

cv2.imshow('colorH', result)
cv2.waitKey(0)




