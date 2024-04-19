# -*- coding: utf-8 -*-
'''@Time: 2024/4/6 21:50

'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
#方法1：灰度图像直方图均衡化
# 获取灰度图像
img = cv2.imread("../lenna.png",1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("image_gray",gray)
# cv2.waitKey(2)
#灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

#直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

#把灰度图和均衡化图对比
cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))
cv2.waitKey(0)

#彩色图像直方图均衡化
#彩色图像均衡化，需要分解通道 对每个通道均衡化
# (b,g,r) = cv2.split(img)
# bH = cv2.equalizeHist(b)
# gH = cv2.equalizeHist(g)
# rH = cv2.equalizeHist(r)
# #合并每个通道
# result = cv2.merge((bH,gH,rH))
# cv2.imshow("dst_rgb",np.hstack([img,result]))
# cv2.waitKey(0)