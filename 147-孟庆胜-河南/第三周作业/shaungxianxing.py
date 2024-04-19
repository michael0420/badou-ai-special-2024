#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2




def chazhi(img, out_dim):
    src_h, src_w, channel = img.shape
    new_h, new_w = out_dim[1], out_dim[0]
    print('src_h, src_w = ', src_h, src_w)
    print('new_h, new_w = ', new_h, new_w)
    if src_h == new_h and src_w == new_w:
        return img.copy()
    new_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / new_w, float(src_h) / new_h
    for i in range(channel):
        for new_y in range(new_h):
            for new_x in range(new_w):
                # 确认中心点坐标
                src_x = (new_x + 0.5) * scale_x - 0.5
                src_y = (new_y + 0.5) * scale_y - 0.5
                # 防止出现数据越界
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                # 计算虚拟点的像素值
                value1 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                value2 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                new_img[new_y, new_x, i] = int((src_y1 - src_y) * value1 + (src_y - src_y0) * value2)

    return new_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    new = chazhi(img, (300, 300))
    cv2.imshow('bilinear interp', new)
    cv2.waitKey()


