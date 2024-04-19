# coding = utf-8

'''
    Sobel边缘检测
'''

import cv2
import numpy as np

img = cv2.imread('lenna.png')

# 调用接口，转为有符号16位数
x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 转回uint8图像
X = cv2.convertScaleAbs(x)
Y = cv2.convertScaleAbs(y)
cv2.imshow('x', X)
cv2.imshow('y', Y)

# 组合x、y方向的边缘检测结果
ok = cv2.addWeighted(X, 0.5, Y, 0.5, 0)
cv2.imshow('final', ok)
cv2.waitKey(0)
