#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np

img = cv2.imread("lenna.png", 0)

# 分别检测水平和垂直边缘
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1)
# cv2.imshow("absX", sobel_x)
# cv2.imshow("absY", sobel_y)

# 将检测到到结果转换为uint8数据形式，增强显示效果
absX = cv2.convertScaleAbs(sobel_x)
absY = cv2.convertScaleAbs(sobel_y)

# 两张图像权重相加
dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

cv2.imshow("absX", absX)
cv2.imshow("absY", absY)

cv2.imshow("dst_img", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()