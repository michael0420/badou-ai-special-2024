#!/usr/bin/env python
#coding=utf-8

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('E:\image\lenna.png')

# Sobel 函数求导 先扩大范围在缩小范围
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

'''
在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。
否则将无法显示图像，而只是一副灰色的窗口。
dst = cv2.convertScaleAbs(src[, dst[, alpha[, beta]]])  
其中可选参数alpha是伸缩系数，beta是加到结果上的一个值。结果返回uint8类型的图片。
'''

absx = cv2.convertScaleAbs(x)
absy = cv2.convertScaleAbs(y)

'''
 进行的是两个方向的求导 要进行合并
用cv2.addWeighted(...)函数将其组合起来
dst = cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])  
其中alpha是第一幅图片中元素的权重，beta是第二个的权重，
gamma是加到最后结果上的一个值。
'''

dst = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)

plt.subplot(221)
plt.imshow(absx)

plt.subplot(222)
plt.imshow(absy)

plt.subplot(223)
plt.imshow(dst)
plt.show()

cv2.imshow("absx", absx)
cv2.imshow("absy", absy)
cv2.imshow("Result", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()