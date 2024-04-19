# -*- coding: utf-8 -*-
"""
@File    :   sobel-1.py
@Time    :   2024/04/08 20:46:52
@Author  :   廖红洋 
"""

import cv2
import numpy as np

img = cv2.imread("lenna.png", 0)  # 单通道灰度图

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 横向卷积
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 纵向卷积

absX = cv2.convertScaleAbs(x)  # 转换为unit8类型
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 组合横纵向边缘

cv2.imwrite("absX.png", absX)  # 保存三个结果
cv2.imwrite("absY.png", absY)
cv2.imwrite("Result.png", dst)
