#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np


#读取灰度图
img = cv2.imread("lenna.png", cv2.IMREAD_GRAYSCALE)

#运用sobel对图像信息求导获得x和y方向的导数值
x = cv2.Sobel(img,cv2.CV_16S, 1,0 )
y = cv2.Sobel(img,cv2.CV_16S, 0,1 )


#将以上获得的值转化为可显示为图像的形式

abs_x = cv2.convertScaleAbs(x)
abs_y = cv2.convertScaleAbs(y)

#将x和y方向的计算整合起来，获得完整的图像

dst = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)


#展示
cv2.imshow('X', abs_x)
cv2.imshow('Y', abs_y)
cv2.imshow('Total', dst)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


#直方图均衡化

import cv2
import numpy as np
from matplotlib import pyplot as plt

#读取图像灰度信息
img = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE) 
#利用opencv中eaqualizeHist代码对图像进行均衡化处理
equ = cv2.equalizeHist(img)

#利用matplot函数展示出前后图像和直方图
plt.subplot(221),plt.imshow(img, "gray"),plt.title("img"), plt.xticks([]),plt.yticks([])
plt.subplot(222),plt.imshow(equ, "gray"),plt.title("equ"), plt.xticks([]),plt.yticks([])
plt.subplot(223),plt.hist(img.ravel(),256),plt.title("img_hist")
plt.subplot(224),plt.hist(equ.ravel(),256),plt.title("equ_hist")


cv2.waitKey(0)
cv2.destroyAllWindows()

