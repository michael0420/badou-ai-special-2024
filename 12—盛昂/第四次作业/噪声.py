#!/usr/bin/env python
# coding: utf-8

# In[30]:


# 椒盐噪声  
import numpy as np
import cv2
from numpy import shape
import random

def saltpepper(src,percentage):
    noiseImg =src
    nums =int(percentage*src.shape[0]*src.shape[1])
    #每次取一个随机点，生成随机行、列
    for i in range(nums):
        randX =random.randint(0,src.shape[0]-1)
        randY =random.randint(0,src.shape[1]-1)
    
        if random.random()<=0.5:
            noiseImg[randX,randY] =0
        else:
            noiseImg[randX,randY]= 255
        return noiseImg

img =cv2.imread("lenna.png",0)
speImg =saltpepper(img,0.8)
img =cv2.imread("lenna.png")
img1 =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("srcImg",img1)
cv2.imshow("speImg",speImg)
cv2.waitKey(0)


# In[34]:


# 高斯噪声
import numpy as np
import cv2
from numpy import shape
import random 

def funcGuass(src,means,sigma,percentage):
    noiseImg =src
    nums =int(percentage*src.shape[0]*src.shape[1])
#    遍历图像元素噪声点
    for i in range(nums):
        #生成随机的行列
        randX =random.randint(0,src.shape[0]-1)
        randY =random.randint(0,src.shape[1]-1)
        
        noiseImg[randX,randY] =noiseImg[randX,randY]+random.gauss(means,sigma)
        
        if noiseImg[randX,randY]<0:
            noiseImg[randX,randY] =0
        elif noiseImg[randX,randY]>255:
            noiseImg[randX,randY]=255
            
    return noiseImg

img =cv2.imread("lenna.png")
img1 =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gusImg =funcGuass(img1,2,4,0.8)
cv2.imshow("src",img1)
cv2.imshow("gusImg",gusImg)
cv2.waitKey(0)
    


# In[ ]:


# 调用接口的形式实现噪声图
import numpy as np
import cv2
from PIL import Image
from skimage import util

img =cv2.imread("lenna.png")
noise_sp_img =util.random_noise(img,mode ='s&p')
noise_gs_img=util.random_noise(img,mode ="gaussian")

cv2.imshow("source",img)
cv2.imshow("S&P",noise_sp_img)
cv2.imshow("Gauss",noise_gs_img)
cv2.waitKey(0)
cv2.destoryAllWindow()

