# -*- coding: utf-8 -*-
'''
@ Author: Hayne
@ Time: 2024/04/04
@ Name: sobel
'''

import cv2

img = cv2.imread("lenna.png", 0)

'''
边缘检测函数：cv2.Sobel()
sobel函数的结果有可能有负值，和大于255的值
因为原图是unit8，即8位无符号数（[0,255]），需要转化为16位有符号的数据类型
需要在参数中设置cv2.CV_16S
'''
X = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 水平sobel卷积核，求垂直边界
Y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 垂直sobel卷积核，求水平边界

# 用cv2.convertScaleAbs()函数将其转回原来的uint8形式
x = cv2.convertScaleAbs(X)
y = cv2.convertScaleAbs(Y)

# 用cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])函数将两个方向的结果组合起来
# 其中alpha是第一幅图片中元素的权重，beta是第二个的权重，gamma是加到最后结果上的一个值。
out = cv2.addWeighted(x, 0.5, y, 0.5, 0)

cv2.imshow("X", x)
cv2.imshow("Y", y)

cv2.imshow("Result", out)

cv2.waitKey(0)
