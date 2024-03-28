# -*- coding: utf-8 -*-
"""
@File    :   interp.py
@Time    :   2024/03/27 16:30:53
@Author  :   廖红洋 
"""

import cv2
import numpy as np

img = cv2.imread("lenna.png")
h, w, chan = img.shape
h_new, w_new = 800, 800  # 放大后的图片大小(单位像素)
img_interp = np.zeros((h_new, w_new, chan), np.uint8)  # 数据类型为unit8(0-255)
for i in range(h_new):
    for j in range(w_new):
        x_near = int(i * h / h_new + 0.5)  # h/h_new为相对比例，乘以i就得到原图上最近的点的坐标
        y_near = int(j * w / w_new + 0.5)
        img_interp[i, j] = img[x_near, y_near]

cv2.imshow("nearest interp", img_interp)  # 显示图片
cv2.imwrite("lenna_interp.png", img_interp)  # 保存
cv2.waitKey(0)  # 窗口保持打开，否则图片会关闭
