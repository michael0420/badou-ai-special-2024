# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:14:10 2024

@author: DELL
## 双线性插值
"""

import cv2
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape  # 获取原图高，宽，通道数
    dst_h, dst_w = out_dim[1], out_dim[0]  # 获取目标图像高，宽
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:  # 判断目标图像大小如果与原图一致，则直接复制原图
        return img.copy()
    dst_image = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 缩放比例
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5  # 使原图和目标图中心对称，两边加偏移z=0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5  # 先找到目标图像对应原图像的虚拟点坐标
                # 对超出边界值的处理
                # src_x0和src_x1分别对应公式里的x1,x2
                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)  # 数组索引是从0开始的，所以有效索引范围从0到src_w-1
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 代入双线性插值公式(先在x方向做插值，后y）
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]  # f(R1)
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]  # f(R2)
                dst_image[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)  # 先高后宽
    return dst_image


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img, (1000, 1000))
    cv2.imshow('original image', img)
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey(0)
