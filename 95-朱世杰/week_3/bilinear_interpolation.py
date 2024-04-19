"""
双线性插值法
@Author: zsj
"""
import numpy as np
import cv2


# 双线性插值
def bilinear_interpolate(img, dst_h, dst_w):
    src_h, src_w, channel = img.shape
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    img_dest = np.zeros([dst_h, dst_w, channel], np.uint8)
    h_ratio = src_h / dst_h
    w_ratio = src_w / dst_w
    for dst_row in range(dst_h):
        for dst_col in range(dst_w):
            # 获取目标图在原图坐标系（基于几何中心对齐）上的坐标
            src_row = (dst_row + 0.5) * h_ratio - 0.5
            src_col = (dst_col + 0.5) * w_ratio - 0.5
            # 获取该点周围四点的行列数
            row0 = int(src_row)
            row1 = min(row0 + 1, src_h - 1)
            col0 = int(src_col)
            col1 = min(col0 + 1, src_w - 1)
            # 通过该四点坐标计算channel对应的各个色值
            col_1 = (col1 - src_col)
            col_0 = (src_col - col0)
            for color in range(channel):
                temp0 = col_1 * img[row0, col0, color] + col_0 * img[row0, col1, color]
                temp1 = col_1 * img[row1, col0, color] + col_0 * img[row1, col1, color]
                img_dest[dst_row, dst_col, color] = int((row1 - src_row) * temp0 + (src_row - row0) * temp1)
    return img_dest


img = cv2.imread('../lenna.png')
new_img = bilinear_interpolate(img, 900, 900)
cv2.imshow('old img', img)
cv2.imshow('new img', new_img)
cv2.waitKey(0)
