# -*- coding: utf-8 -*-
'''@Time: 2024/1/31 21:37
上采样2
'''

import cv2
import numpy as np


def function(img):
    height,width,channels = img.shape
    emptyImage = np.zeros((800,800,3),np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(height*i/800+0.5)
            y = int(width*j/800+0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage

img = cv2.imread("../lenna.png")
print(img.shape)
zoom = function(img)
print(zoom)
print(zoom.shape)


def bilinear_interpolation(img,out_dim):
    src_h,src_w,channel = img.shape
    dst_h,dst_w = out_dim[1],out_dim[0]
    print(f"src_h={src_h},src_w={src_w}")
    print(f"drc_h={dst_h},drc_w={dst_w}")

    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros([dst_h,dst_w,3],dtype=np.uint8)
    scale_x,scale_y = float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #1.先加0.5，求目标图在原图对应的坐标点
                src_x = (dst_x+0.5) * scale_x - 0.5
                src_y = (dst_y+0.5) * scale_y - 0.5
                #2.防超出原图坐标
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0+1,src_w-1)
                src_y1 = min(src_y0+1,src_h-1)
                #3.计算
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0)



cv2.imshow("nearest intert",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)