'''
sobel 边缘检测
'''

import numpy as np
import cv2

# 卷积: direction: 1水平 2垂直 3全部
def function(img,direction=3):
    # 获取图像的灰度图像
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 使用 Sobel 计算水平和垂直方向的梯度
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    if (direction == 1):
        return sobel_x
    elif direction == 2:
        return sobel_y
    else:
        # 将结果转换位uint8类型，并取绝对值
        sobel_x = np.uint8(np.absolute(sobel_x))
        sobel_y = np.uint8(np.absolute(sobel_y))
        # 合并水平和垂直方向的sobel边缘检测结果
        sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)
        return sobel_combined

img = cv2.imread('lenna.png')
zoom = function(img)
cv2.imshow('image', img)
# 显示边缘检测结果
cv2.imshow('sobel edge detection', zoom)
cv2.waitKey(0)
