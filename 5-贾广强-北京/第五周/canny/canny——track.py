#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2
import numpy as np
def CannyThreshold(lowThreshold):
    detected_edges = cv2.Canny(gray,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size) 
    dst = cv2.bitwise_and(img,img,mask= detected_edges)
    cv2.imshow('canny_result',dst)

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3
    
img = cv2.imread('lenna.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('canny_result')
    
cv2.createTrackbar('Min threshold','canny_result',lowThreshold, max_lowThreshold,CannyThreshold)
CannyThreshold(0)
if cv2.waitKey(0)==27:
    cv2.destroyAllWindows()
    

