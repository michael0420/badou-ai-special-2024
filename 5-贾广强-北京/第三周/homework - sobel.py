#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np

img = cv2.imread('lenna.jpg',0)
sobel_x = cv2.Sobel(img,cv2.CV_16S,1,0)
sobel_y = cv2.Sobel(img,cv2.CV_16S,0,1)
abs_x = cv2.convertScaleAbs(sobel_x)
abs_y = cv2.convertScaleAbs(sobel_y)
dst = cv2.addWeighted(abs_x,0.5,abs_y,0.5,0)
cv2.imshow('abs_x',abs_x)
cv2.imshow('abs_y',abs_y)
cv2.imshow('result',dst)
cv2.waitKey()

