# _*_ coding: UTF-8 _*_
# @Time: 2024/4/17 16:02
# @Author: iris
import cv2
import numpy as np

img = cv2.imread('../data/lenna.png')
# 均值滤波
dst_1 = cv2.blur(img, (5, 5))
# 高斯滤波
dst_2 = cv2.GaussianBlur(img, (5, 5), sigmaX=0.3)
# 中值滤波
dst_3 = cv2.medianBlur(img, 5)

cv2.imshow('img', img)
cv2.imshow('dst_1', dst_1)
cv2.imshow('dst_2', dst_2)
cv2.imshow('dst_3', dst_3)
cv2.waitKey(0)
