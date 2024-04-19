#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
 

def bilinear_interpolation(img,out_dim):
    old_h, old_w, old_s = img.shape
    new_h, new_w = out_dim[1], out_dim[0]
    print ("old_h, old_w = ", old_h, old_w)
    print ("new_h, new_w = ", new_h, new_w)
    if old_h == new_h and old_w == new_w:
        return img.copy()
    # 缩放比例
    scale_h = float(old_h) / new_h
    scale_w = float(old_w) / new_w
    # 创建新图片
    new_img = np.zeros((new_h,new_w,3),dtype=np.uint8)
    for i in range(old_s):
        for x in range(new_h):
            for y in range(new_w):
                # 用对称中心法找到x,y对应的原图的坐标
                old_x = (x + 0.5) * scale_h - 0.5
                old_y = (y + 0.5) * scale_w - 0.5

                # 找出附件的四个点
                # old_x0 = max(0, int(old_x - 1))
                old_x0 = int(np.floor(old_x))
                old_x1 = min(old_x0 + 1, old_h)
                # old_y0 = max(0, int(old_y - 1))
                old_y0 = int(np.floor(old_y))
                old_y1 = min(old_y0, old_w)

                #代入公式计算
                t1 = int(old_y1 - y) * ((old_x1 - x) * img[old_x0, old_y0, i] + (x - old_x1) * img[old_x1, old_y0, i])
                t2 = int(y - old_y0) * ((old_x1 - x) * img[old_x0, old_y1, i] + (x - old_x1) * img[old_x1, old_y1, i])
                new_img[x,y,i] = int(t1 + t2)
    return new_img
 
 
if __name__ == '__main__':
    img = cv2.imread('scaled_binary_image.jpg')
    new = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear_interp',new)
    cv2.waitKey()