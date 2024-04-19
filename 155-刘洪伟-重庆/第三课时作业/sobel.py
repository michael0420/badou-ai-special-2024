# _*_ coding: UTF-8 _*_
# @Time: 2024/4/3 19:51
# @Author: iris
import cv2

image = cv2.imread('../data/lenna.png')
x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
y = cv2.Sobel(image, cv2.CV_16S, 0, 1)

abs_x = cv2.convertScaleAbs(x)
abs_y = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
cv2.imshow('abs_x', abs_x)
cv2.imshow('abs_y', abs_y)
cv2.imshow('result', dst)
cv2.waitKey(0)
