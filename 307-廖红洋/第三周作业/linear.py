# -*- coding: utf-8 -*-
"""
@File    :   linear.py
@Time    :   2024/04/07 10:23:52
@Author  :   廖红洋 
"""

import numpy as np
import cv2

img = cv2.imread("lenna.png")
h, w, chan = img.shape
fh, fw = 1000, 1000  # 输出尺寸
dst_img = np.zeros((fh, fw, 3), dtype=np.uint8)
for i in range(chan):
    for y in range(fh):
        for x in range(fw):
            y_old = (y + 0.5) * h / fh - 0.5  # 得到对应原图的最近坐标,加0.5是为了让几何中心对齐
            x_old = (x + 0.5) * w / fw - 0.5
            src_x0 = int(np.floor(x_old))  # 取最临近的四个点做二线性插值
            src_x1 = min(src_x0 + 1, w - 1)
            src_y0 = int(np.floor(y_old))
            src_y1 = min(src_y0 + 1, h - 1)
            temp0 = (src_x1 - x_old) * img[src_y0, src_x0, i] + (
                x_old - src_x0
            ) * img[  # 计算该点插值
                src_y0, src_x1, i
            ]
            temp1 = (src_x1 - x_old) * img[src_y1, src_x0, i] + (x_old - src_x0) * img[
                src_y1, src_x1, i
            ]
            dst_img[y, x, i] = int((src_y1 - y_old) * temp0 + (y_old - src_y0) * temp1)
cv2.imshow("二线性插值", dst_img)
cv2.waitKey()
