#!/usr/bin/env python
# coding: utf-8

# In[19]:


import cv2
import numpy as np 
from skimage.color import rgb2gray
from PIL import Image
import matplotlib.pyplot as plt

plt.subplot(221)
my_img = plt.imread('lenna.jpg')
plt.imshow(my_img)

my_img = cv2.imread('lenna.jpg')
img_gray = rgb2gray(my_img)
# cv2.imshow('img_gray',img_gray)
# cv2.waitKey(0)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

my_img_binary = np.where(img_gray >= 0.5,1,0)

plt.subplot(224)
plt.imshow(my_img_binary,cmap = 'gray')
plt.show()

