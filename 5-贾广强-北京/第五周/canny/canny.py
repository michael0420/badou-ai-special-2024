#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
img = cv2.imread('lenna.jpg',0)
cv2.imshow('canny',cv2.Canny(img,50,100))
cv2.waitKey()
cv2.destroyAllWindows()


# In[5]:




