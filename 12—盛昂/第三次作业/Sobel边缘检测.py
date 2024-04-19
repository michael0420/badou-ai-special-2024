#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np

img1 =cv2.imread('lenna.png')
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#由uint8无符号转换为uint16有符号，增加图像位数，
x =cv2.Sobel(img,cv2.CV_16S, 1, 0)
y =cv2.Sobel(img,cv2.CV_16S, 0, 1)

# 经过处理后，用convertScaleAbs()函数将其转换为原来的uint8形式，否则无法显示图像，

absX =cv2.convertScaleAbs(x)
absY =cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX,0.5,absY,0.5,0)

# cv2.imshow("absX",absX)
# cv2.imshow("absY",absY)

# cv2.imshow("Result",dst)
cv2.imshow("imgSobel", np.hstack([absX,absY,dst]))
cv2.waitKey(0)


# In[ ]:




