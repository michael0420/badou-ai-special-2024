#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像，灰度化
img =cv2.imread("lenna.png")
gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 均值化
dst =cv2.equalizeHist(gray)

# # 生成直方图
hist =cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(),256)
plt.show()

# 打印图像
cv2.imshow("EqualizationHist",np.hstack([gray,dst]))
cv2.waitKey()


(b,g,r) =cv2.split(img)
bH =cv2.equalizeHist(b)
gH =cv2.equalizeHist(g)
rH =cv2.equalizeHist(r)

# 合并为一个通道
result =cv2.merge((bH,gH,rH))
cv2.imshow("dst_rgb",np.hstack([img,result]))
cv2.waitKey()

