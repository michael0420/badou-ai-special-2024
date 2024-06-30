#!/usr/bin/env python
# coding: utf-8

# In[33]:


import cv2
import numpy as np

def bilinear_interpolation(scale_height, scale_width, img):
    
    #读取图像信息，创建新空白图像
    height, width, channels = img.shape
    scale_image = np.zeros((scale_height, scale_width, channels), dtype = np.uint8)
    
    #如果新图像的高原与原始图像相等，返回原始图像
    if scale_height == height and scale_width == width:
        scale_image = img
        return scale_image
    else:
        for k in range (channels):
         for i in range (scale_height):
            for j in range (scale_width):
                
                #对齐中心      
                x = ( i + 0.5)*(height/scale_height) - 0.5
                y = ( j + 0.5)*(width/scale_width) - 0.5
                
                #双线性函数计算 首先在x方向进行两次插值，再在y方向上进行一次插值
                #计算出相邻四个点的位置
                x1, y1 = int (np.floor(x)), int(np.floor(y))
                x2, y2 = min (x1+1, height-1), min(y1 + 1, width - 1) #防止溢出
                
                p11 = img [x1,y1,k]
                p12 = img [x1,y2,k]
                p21 = img [x2,y1,k]
                p22 = img [x2,y2,k]
                
                #插值公式
                R1 = (x-x1)*p11 +(x2-x)*p21
                R2 = (x-x1)*p12 +(x2-x)*p22
                scale_image[i,j,k] = (y-y1)*R1 +(y2-y)*R2
                
    return scale_image
            
if __name__ == '__main__':           
    img = cv2.imread("lenna.png")
    new_img = bilinear_interpolation(800,800, img)

    cv2.imshow("new pic", new_img)
    cv2.waitKey(0)


# In[ ]:




