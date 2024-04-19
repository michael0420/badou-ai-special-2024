#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

'''
双线性插值实现
'''

'''
img:原始图像
out_img:目标图像
'''


def double_line(img, out_img):
    src_h, src_w, channel = img.shape  # 原始图像高，宽，通道数
    out_h, out_w = out_img[1], out_img[0]  # 目标图像高，宽

    # 如果原始图像高和宽与目标图像高和宽相同，则直接返回原始图像的拷贝即可
    if src_h == out_h and src_w == out_w:
        return img.copy()

    #  初始化目标图像
    out_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    #  计算原始图像转换为目标图像，高和宽需要调整的比例
    scale_x, scale_y = float(src_w) / out_w, float(src_h) / out_h

    #  循环通道数
    for i in range(channel):
        #  循环目标图像高度
        for out_y in range(out_h):
            #  循环目标图像宽度
            for out_x in range(out_w):
                #  使用几何中心对称，找到目标图像的x,y值
                src_x = (out_x + 0.5) * scale_x - 0.5
                src_y = (out_y + 0.5) * scale_y - 0.5

                #  找到用于计算插值的点的坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 计算差值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]

                out_img[out_y, out_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return out_img


if __name__ == '__main__':
    # 读入图像
    img = cv2.imread('lenna.png')
    # 通过双线性插值返回目标图像
    out = double_line(img, (700, 700))
    # 输出目标图像
    cv2.imshow('double_line ', out)
    cv2.waitKey()
