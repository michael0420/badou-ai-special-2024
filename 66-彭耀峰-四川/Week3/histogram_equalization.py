"""
实现直方图均衡化
"""

import cv2
from matplotlib import pyplot  as plt
import numpy as np


#灰度图像直方图均衡化-单通道
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)

hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

cv2.imshow("Histogram equalization",np.hstack([gray,dst]))
cv2.waitKey(0)



# # 彩色图像直方图均衡化
# img = cv2.imread("lenna.png", 1)
# cv2.imshow("src", img)
#
# # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# (b, g, r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# # 合并每一个通道
# result = cv2.merge((bH, gH, rH))
# cv2.imshow("dst_rgb", result)
#
# cv2.waitKey(0)
