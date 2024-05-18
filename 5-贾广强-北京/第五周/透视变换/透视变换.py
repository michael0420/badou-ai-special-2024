#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
img = cv2.imread('photo1.jpg')
img_copy = img.copy()

src = np.float32([[207,151],[517,285],[17,601],[343,731]])
dst = np.float32([[0,0],[337,0],[0,488],[337,488]])
print(img.shape)
m= cv2.getPerspectiveTransform(src,dst)
print('waroMatrix:')
print(m)
result = cv2.warpPerspective(img_copy,m,(337,488))
cv2.imshow('src',img_copy)
cv2.imshow('dst',result)
cv2.waitKey(0)
cv2.destroyAllWindows()

