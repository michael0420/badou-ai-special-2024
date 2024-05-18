#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from numpy import shape
import random
from matplotlib import pyplot as plt

def Gaussian_img(img, percentage, mean, sigma ):
    Gaussian_img = np.copy(img)
    height, width= Gaussian_img.shape[:2]
    num = int(percentage*height*width)
    for i in range(num):

        random_x = random.randint(0, height - 1)
        random_y = random.randint(0, width - 1)
        #Pout = pin + random.gauss 通过sigma和mean来生成符合高斯分布的函数
        for c in range(Gaussian_img.shape[2]):
            Gaussian_img[random_x,random_y,c] = Gaussian_img[random_x,random_y, c] + random.gauss(mean,sigma)
            if Gaussian_img[random_x,random_y,c] < 0:
                Gaussian_img[random_x, random_y,c] = 0
            elif Gaussian_img[random_x,random_y,c] > 255:
                Gaussian_img[random_x, random_y,c] = 255    
                    
    return Gaussian_img

img = cv2.imread('lenna.png',cv2.IMREAD_COLOR)

Gaussianimg1 = Gaussian_img(img, 0.05, 5, 15)
Gaussianimg2 = Gaussian_img(img, 0.8, 5, 15)
Gaussianimg3 = Gaussian_img(img, 0.8, 0, 3)
Gaussianimg4 = Gaussian_img(img, 0.05, 0, 3)

plt.subplot(221), plt.imshow(cv2.cvtColor(Gaussianimg1, cv2.COLOR_BGR2RGB)), plt.title("low percentage & high value"),plt.xticks([]),plt.yticks([])
plt.subplot(222), plt.imshow(cv2.cvtColor(Gaussianimg2, cv2.COLOR_BGR2RGB)), plt.title("high percentage & high value"),plt.xticks([]),plt.yticks([])
plt.subplot(223), plt.imshow(cv2.cvtColor(Gaussianimg3, cv2.COLOR_BGR2RGB)), plt.title("high percentage & low value"),plt.xticks([]),plt.yticks([])
plt.subplot(224), plt.imshow(cv2.cvtColor(Gaussianimg4, cv2.COLOR_BGR2RGB)), plt.title("low percentage & low value"),plt.xticks([]),plt.yticks([])

cv2.waitKey(0)


# In[ ]:




