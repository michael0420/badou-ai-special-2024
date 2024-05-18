#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

img=cv2.imread('lenna.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray",gray)

# 灰度图像直方图均衡化
dst=cv2.equalizeHist(gray)

#直方图
hist=cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(),256) #plt.hist(src,pixels)绘制直方图,src:数据源，注意这里只能传入一维数组，使用src.ravel()可以将二维图像拉平为一维数组,pixels:像素级，一般输入256。
plt.show()
cv2.imshow("Histogram Equalization",np.hstack([gray,dst]))


#彩色图像直方图均衡化
# img=cv2.imread('lenna.png',1)
# cv2.imshow('src',img)
#
# #彩色图像均衡化，需要分解通道，对每一个通道均衡化
# (b,g,r)=cv2.split(img)
# bH=cv2.equalizeHist(b)
# gH=cv2.equalizeHist(g)
# rH=cv2.equalizeHist(r)
# result=cv2.merge([bH,gH,rH])    #函数 cv2.merge() 将 B、G、R 单通道合并为 3 通道 BGR 彩色图像。
# cv2.imshow('dst_rgb',result)






cv2.waitKey(0)