#!/usr/bin/env python
# coding: utf-8

# In[16]:


#灰色直方图均衡化
import cv2
import numpy as np
import matplotlib.pyplot as plt

#获取原始图像
o_img = cv2.imread('lenna.jpg',1)
#灰度化
g_img = cv2.cvtColor(o_img,cv2.COLOR_BGR2GRAY)
# 计算原始图像直方图
o_hist = cv2.calcHist([o_img],[0],None,[256],[0,256])
plt.figure()
plt.hist(g_img.ravel(),256)
plt.show()
#直方图均衡化
eq_img = cv2.equalizeHist(g_img)
#计算直方图
d_hist = cv2.calcHist([eq_img],[0],None,[256],[0,256])
plt.figure()
plt.hist(eq_img.ravel(),256)
plt.show()

cv2.imshow('gray vs Histogram',np.hstack([g_img,eq_img]))
cv2.waitKey()


# In[3]:


#彩色直方图均衡化
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lenna.jpg',1)
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
result = cv2.merge((bH,gH,rH))
cv2.imshow('vs',np.hstack([img,result]))
cv2.waitKey()

