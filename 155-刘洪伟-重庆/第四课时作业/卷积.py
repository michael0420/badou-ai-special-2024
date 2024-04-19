# _*_ coding: UTF-8 _*_
# @Time: 2024/4/17 16:00
# @Author: iris
import cv2
import numpy as np

img = cv2.imread('../data/lenna.png')
# 相当于原始图片中的每个点都被平均了一下，所以图像变模糊了
# kernel = np.ones((5, 5), np.float32) / 25
kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
dst = cv2.filter2D(img, -1, kernel)
cv2.imshow('image', np.hstack((img, dst)))
cv2.waitKey(0)
