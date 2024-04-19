#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import cv2
# 读取图像
img =cv2.imread("lenna.png")
# 获取图像的尺寸
height,width = img.shape[:2]
# 定义一个和原来一样尺寸的新图像，注意是方括号
img_gray = np.zeros([height,width],img.dtype)
# 遍历每一个元素
for i in range(height):
    for j in range(width):
        m = img[i,j]
        #用公式将元素灰度化
        img_gray[i,j] = m[0]*0.11 +m[1]*0.59 +m[2]*0.3
# 显示图像
cv2.imshow("img_gray",img_gray)
# 要加这个哦
cv2.waitKey(0)

