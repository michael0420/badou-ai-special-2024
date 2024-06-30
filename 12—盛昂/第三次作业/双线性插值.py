#!/usr/bin/env python
# coding: utf-8

# In[16]:


# import
import numpy as np
import cv2
# 定义函数
def bilinear_interpolation(img,out_dim):   #outdim：外型尺寸
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1],out_dim[0]
    if src_w == dst_w and src_h == dst_h:
        return img.copy()
    dst_img = np.zeros((dst_w,dst_h,3),dtype = np.uint8)
    scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h
    
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                
            #做一个边界，以免超过边界
            #取中心坐标
                src_x = (dst_x+0.5) *scale_x -0.5
                src_y = (dst_y+0.5) * scale_y -0.5
            #做一个边界
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 +1, src_w -1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 +1, src_h -1)
                
            #计算双线性插值
                temp0 = (src_x1- src_x)*img[src_x0, src_y0, i] + (src_x -src_x0)*img[src_x1, src_y0, i]
                temp1 = (src_x1 -src_x)*img[src_x0,src_y1, i] + (src_x -src_x0)*img[src_x1, src_y1, i]
                dst_img[dst_x, dst_y, i] = int((src_y1-src_y)*temp0 + (src_y -src_y0)*temp1)
                
    return dst_img

if __name__ == "__main__":
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()

