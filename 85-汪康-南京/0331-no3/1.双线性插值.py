# -*- coding: utf-8 -*-
'''@Time: 2024/4/6 21:50
python 双线性插值
'''
import numpy as np
import cv2

def bilinear_interpolation(img,out_dim):
    src_h,src_w,channel = img.shape
    dst_h,dst_w = out_dim[1],out_dim[0]
    print("原图高 宽",src_h,src_w)
    print("目标图高 宽",dst_h,dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x,scale_y = float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #两个图像的几何中心重合时，找到原图坐标
                src_x = (dst_x+0.5)*scale_x -0.5
                src_y = (dst_y+0.5)*scale_y -0.5

                #处理一下边界
                src_x0 = int(np.floor(src_x)) #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1,src_h - 1)

                # 套公式计算插值
                R1 = (src_x1 - src_x)*img[src_y0,src_x0,i] + (src_x - src_x0)*img[src_y0,src_x1,i]
                R2 = (src_x1 - src_x)*img[src_y1,src_x0,i] + (src_x - src_x0)*img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int(
                    (src_y1 - src_y)*R1 + (src_y - src_y0)*R2
                )
    return dst_img

if __name__ =='__main__':
    img = cv2.imread("../lenna.png")
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('src', img)
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()