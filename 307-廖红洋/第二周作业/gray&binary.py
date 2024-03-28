# -*- coding: utf-8 -*-
"""
@File    :   gray&binary.py
@Time    :   2024/03/27 16:00:01
@Author  :   廖红洋 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
h, w = img.shape[:2]  # 等号左边是解构赋值，右边[:2]表示此范围内的数据，左闭右开
img_gray = np.zeros((h, w), img.dtype)  # 第一个参数指定维度，第二个参数指定图的类型

for i in range(h):
    for j in range(w):
        img_gray[i, j] = int(
            img[i, j, 0] / 3 + img[i, j, 1] / 3 + img[i, j, 2] / 3
        )  # 这里采用三通道相加除以3的方法

cv2.imwrite("lenna_gray.png", img_gray)  # 保存到当前路径
plt.imshow(img_gray, cmap="gray")  # 如果需要plt打印灰度图，需要将通道指定为"gray"，否则会显示其他颜色

img_binary = np.where(img_gray >= 128, 255, 0)  # 进而进行二值化
cv2.imwrite("lenna_binary.png", img_binary)  # 同样保存到当前路径
plt.imshow(img_binary, cmap="gray")

plt.show()  # 显示图形
