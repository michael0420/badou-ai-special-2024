#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def bilinear_interpolation(img, dst_h, dst_w):
    img_h, img_w, channels = img.shape
    print("src_h, src_w = ", img_h, img_w)
    dst_img = np.zeros((dst_h, dst_w,3), dtype=np.uint8)
    s_h = float(img_h/dst_h)
    s_w = float(img_w/dst_w)

    if img_h == dst_h and img_w == dst_w:
        return img.copy

    for channel in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 目标图与原图中心中心重合
                src_x = (dst_x + 0.5) * s_w - 0.5
                src_y = (dst_y + 0.5) * s_h - 0.5

                # 找到目标点在原图上的四个点坐标
                src_x1 = int(np.floor(src_x))
                src_x2 = min(src_x1 + 1, img_w-1)
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, img_h-1)
                # 双线性差值公式
                f_r1 = (src_x2 - src_x) * img[src_y1, src_x1, channel] + (src_x - src_x1) * img[src_y1, src_x2, channel]
                f_r2 = (src_x2 - src_x) * img[src_y2, src_x1, channel] + (src_x - src_x1) * img[src_y2, src_x2, channel]
                dst_img[dst_y, dst_x, channel] = int((src_y2 - src_y) * f_r1 + (src_y - src_y1) * f_r2)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("69e754ae5b5a3ecf8f150894ee6b095.jpg")
    h = 1500
    w = 1500
    a = bilinear_interpolation(img, h, w)
    cv2.imshow("img", a)
    cv2.waitKey(0)
