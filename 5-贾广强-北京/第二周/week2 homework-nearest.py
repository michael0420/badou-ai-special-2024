#!/usr/bin/env python
# coding: utf-8

# In[10]:


import cv2
import numpy as np
def function(img):
    height,width,channels = img.shape
    emtpyImage = np.zeros((400,400,channels),np.uint8)
    h = 400/height
    w = 400/width
    for i in range(400):
        for j in range(400):
            x = int(i/h + 0.5)
            y = int(j/h + 0.5)
            emtpyImage[i,j]=img[x,y]
    return emtpyImage
img = cv2.imread('lenna.jpg')
zoom = function(img)
cv2.imshow('nearest',zoom)
cv2.imshow('img',img)
cv2.waitKey()

