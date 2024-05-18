#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def bilinear_interpolation(img,out_dim):
    src_h,src_w,channel =img.shape[:]
    dst_h,dst_w=out_dim[0],out_dim[1]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h==dst_h and src_w==dst_w:
        return img.copy()
    dst_img=np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x,scale_y=float(src_w)/dst_w,float(src_h)/dst_h   #缩放比例
    for i in range(channel):                          #RGB三个图层，每个图层的像素值都要做一次双线性插值，遍历
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x=(dst_x+0.5)* scale_x-0.5     #中心重合法：求新图某个点坐标对应到原图时的坐标值坐标值
                src_y=(dst_y+0.5)*scale_y-0.5

                # 找出用于计算插值的四个相邻点的坐标
                src_x1=int(np.floor(src_x))       #np.floor()返回不大于输入参数的最大整数。（向下取整）,np.floor(2.5)=2.0
                src_x2=min(src_x1+1,src_w-1)      #坐标大小是从1开始的，但是src_h, src_w, channel = img.shape 索引是从0开始的，所以最后一个点坐标应该-1
                src_y1 = int(np.floor(src_y))
                src_y2 = min(src_y1 + 1, src_h - 1)
                temp1=(src_x2-src_x)*img[src_y1,src_x1,i]+(src_x-src_x1)*img[src_y1,src_x2,i]
                temp2=(src_x2-src_x)*img[src_y2,src_x1,i]+(src_x-src_x1)*img[src_y2,src_x2,i]
                dst_img[dst_y,dst_x,i]=int((src_y2-src_y)*temp1)+int((src_y-src_y1)*temp2)

    return dst_img

if __name__=='__main__':        #在当前模块作为程序入口时执行以下代码 ，而在被其它模块引入时不执行以下代码。
    img=cv2.imread('lenna.png')
    dst=bilinear_interpolation(img,(600,590))
    cv2.imshow('lenna.png', img)
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
    src_h1, src_w1, channel = dst.shape[:]
    print("src_h1, src_w1 = ", src_h1, src_w1)

