#!/usr/local/bin/python
# coding=utf-8
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
img = cv2.imread("lenna.png", 0)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# 直方图均衡化的灰度图像
dst = cv2.equalizeHist(img)
# 计算直方图
# cv2.calcHist(images, channels, mask, histSize, ranges)其中：
# 参数1：要计算的原图，以方括号的传入，如：[img]
# 参数2：类似前面提到的dims，灰度图写[0]就行，彩色图B / G / R分别传入[0] / [1] / [2]
# 参数3：要计算的区域，计算整幅图的话，写None
# 参数4：前面提到的bins
# 参数5：前面提到的range
dst_hist = cv2.calcHist([dst],[0],None,[256],[0,256])
# hist, bins= np.histogram(a=img, bins=255, range=[0, 255])

plt.figure()
plt.subplot(221)
plt.imshow(img,cmap='gray')
plt.title('img_gray')

plt.subplot(222)
# 绘制直方图
plt.hist(img.ravel(), 256)
plt.title('img_gray Histogram')

plt.subplot(223)
plt.imshow(dst,cmap='gray')
plt.title('img_gray EqualizeHist')

plt.subplot(224)
# ravel()将数组维度拉成一维数组
plt.hist(dst.ravel(), 256)
plt.title('img_gray_equalizeHist Histogram')

plt.show()
# np.hstack()是numpy中实现数组拼接的函数
cv2.imshow("Histogram Equalization", np.hstack([img, dst]))
cv2.waitKey(0)



# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
# split()函数用于图像BGR通道的分离，分离通道的顺序是B、G、R
(b, g, r) = cv2.split(img)
result = cv2.merge((r,g,b))
# 分别计算bgr通道的直方图
b_hist = cv2.calcHist([b],[0],None,[256],[0,256])
g_hist = cv2.calcHist([g],[0],None,[256],[0,256])
r_hist = cv2.calcHist([r],[0],None,[256],[0,256])

plt.subplot(221)
# 绘制bgr 3个通道的直方图曲线
# plt.plot(b_hist, color='b')
# plt.plot(g_hist, color='g')
# plt.plot(r_hist, color='r')
# plt.title('img_bgr Histogram curve')
plt.imshow(result,cmap='gray')
print("---lenna----")

plt.subplot(222)
# 绘制bgr 3通道的直方图
plt.hist(b.ravel(), bins=256,density=1,facecolor='b',edgecolor='b',alpha=0.75)
plt.hist(g.ravel(), bins=256,density=1,facecolor='g',edgecolor='g',alpha=0.75)
plt.hist(r.ravel(), bins=256,density=1,facecolor='r',edgecolor='r',alpha=0.75)
plt.title('img_bgr Histogram')
# bgr 3通道图像直方图均衡化
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并图像，merge()函数用于合并每一个通道
result_H = cv2.merge((rH, gH, bH))

plt.subplot(223)
# # 计算均衡化后的图像bgr3通道直方图
# bH_hist = cv2.calcHist([bH],[0],None,[256],[0,256])
# gH_hist = cv2.calcHist([gH],[0],None,[256],[0,256])
# rH_hist = cv2.calcHist([rH],[0],None,[256],[0,256])
#绘制均衡化后的图像bgr 3通道直方图曲线
# plt.plot(bH_hist, color='b')
# plt.plot(gH_hist, color='g')
# plt.plot(rH_hist, color='r')
# plt.title('img_equalizeHist_bgr Histogram curve')
plt.imshow(result_H,cmap='gray')
print("---lenna_EqualizeHist----")

plt.subplot(224)
# 绘制均衡化后的图像bgr 3通道的直方图
plt.hist(bH.ravel(), bins=256,density=1,facecolor='b',edgecolor='b',alpha=0.75)
plt.hist(gH.ravel(), bins=256,density=1,facecolor='g',edgecolor='g',alpha=0.75)
plt.hist(rH.ravel(), bins=256,density=1,facecolor='r',edgecolor='r',alpha=0.75)
plt.title('img_equalizeHist_bgr Histogram')
plt.show()
