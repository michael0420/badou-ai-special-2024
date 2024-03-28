"""
@Date    : 2024/3/28 20:03
@Author  : zwx
@Version : 1.0
@Desc    : 上采样 - 临近插值法Demo
"""

import numpy as np
import cv2
import math


def customer_function(img, nh, nw):
    h, w, channels = img.shape
    # print(h, w, channels)  # 512 512 3
    new_img = np.zeros((nh, nw, channels), np.uint8)  # 创建相同通道数画布， np.uint8 ： 指定数据类型为 0 到 255 的正整数
    rate_h = nh / h
    rate_w = nw / w
    # print(rate_h, rate_w) # 1.3671875 1.3671875
    for i in range(nh):
        for j in range(nw):
            # x = int(i / rate_h + 0.5) # 向下取整
            # y = int(j / rate_w + 0.5)
            x = round(i / rate_h)  # 四舍五入
            y = round(j / rate_w)
            new_img[i, j] = img[x, y]

    return new_img


img = cv2.imread('../data/input/lenna.png')
new_img = customer_function(img, 700, 700)
print(new_img.shape)
print(new_img)
cv2.imshow("old_img", img)
cv2.imshow("new_img", new_img)
cv2.waitKey(0)  # 等待函数，关闭图片即结束程序
