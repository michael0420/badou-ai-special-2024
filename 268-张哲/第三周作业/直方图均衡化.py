import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''
gray = cv2.imread('lenna.png',0) #0:以灰度模式加载图片，1：以彩色模式加载图片，-1：包括透明度通道(alpha)
img = cv2.imread('lenna.png',1)
# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
#直方图
hist = cv2.calcHist([gray],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(),256)
plt.show()
cv2.imshow('Histogram Equalization:',np.hstack([gray,dst]))
# cv2.waitKey(0)
# 彩色图像直方图均衡化
(b,g,r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
# 合并每一个通道,hstack:并列展示
result = cv2.merge((bh,gh,rh))
cv2.imshow('img_Histogram Equalization:',np.hstack([img,result]))

#直方图
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
    plt.show()
cv2.waitKey(0)