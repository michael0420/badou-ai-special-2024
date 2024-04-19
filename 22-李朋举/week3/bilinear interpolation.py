#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

'''
python implementation of bilinear interpolation
'''


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape  # 512*512 3通道
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("原图 src_h, src_w = ", src_h, src_w)  # 打印 原图 h,w
    print("目标图 dst_h, dst_w = ", dst_h, dst_w)  # 打印 目标图 h,w
    if src_h == dst_h and src_w == dst_w:  # 判断 如果 缩放图的h,w 和 原图的 h,w 一样, 则直接返回不做处理
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)  # 建立一个全0的3通道图像
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # scale_x, scale_y 为缩放比例
    for i in range(channel):  # 遍历3通道
        for dst_y in range(dst_h):  # dst_y 为 缩放后图像的坐标
            for dst_x in range(dst_w):  # dst_x 为 缩放后图像的坐标
                """
                要通过双线性插值的方法算出dst中每一个像素点的像数值：
                1. 通过dst像素点的坐标对应到src图像当中的坐标 
                2. 然后通过双线性插值的方法算出src相应的坐标值
                    
                    1. 通过dst像素点的坐标对应到src图像当中的坐标 
                    for:循环目标图像中的每一个像素    
                    例： 假设目标图dst_x    
                        缩放scale_x        float(src)/dst            [scale=src/dst]
                        原图src_x          src_x = dst_x * scale_x   [dst * scale]   
                        为使 几何中心重合 公式如下
                                          src_x = (dst_x + 0.5) * scale_x-0.5
                """
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                """
                    2. 然后通过双线性插值的方法算出src相应的坐标值
                """
                # find the coordinates of the points which will be used to compute the interpolation
                # 4个坐标
                src_x0 = int(np.floor(src_x))  # np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)  # 防呆 避免超出边界 在算出的src_x0 和 原图像src_w 中取最小值
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                # 将上面4个坐标及f(Q)的值  代入双线性插值算法公式 求出其RGB值    其中 img[src_y0,src_x0,i]为f(Q11) ...
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('D:\cv_workspace\picture\lenna.png')  # 读取彩色原图 三通道
    dst = bilinear_interpolation(img, (700, 700))  # out_dim 定义放大尺寸 原尺寸512*512
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey(1000)
