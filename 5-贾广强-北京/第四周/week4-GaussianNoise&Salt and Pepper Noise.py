#!/usr/bin/env python
# coding: utf-8

# In[17]:


#高斯噪声
import numpy as np
import cv2
from numpy import shape
import random


def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        NoiseImg[randX,
                 randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    print(NoiseImg)
    return NoiseImg


img = cv2.imread('lenna.jpg', 0)
gau_img = GaussianNoise(img, 2, 4, 0.8)
gary_img = cv2.imread('lenna.jpg', 0)
cv2.imshow('gary VS gau', np.hstack([gary_img, gau_img]))
print(gary_img)
cv2.waitKey()


# In[27]:


#椒盐噪声
import numpy as np
import cv2
import random
from numpy import shape

def fun(src,percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0]-1)
        randY = random.randint(0,src.shape[1]-1)
        
        if random.random() <=0.5:
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX,randY] = 255
    return NoiseImg

img = cv2.imread('lenna.jpg',0)
s_p_img = fun(img,0.2)
gary_img = cv2.imread('lenna.jpg',0)
cv2.imshow('gray VS s_p',np.hstack([gary_img,s_p_img]))
cv2.waitKey()


# In[28]:


#使用util.random_noise函数 生成噪声
import cv2
import numpy as np
from skimage import util
# from PIL import  Image
img = cv2.imread('lenna.jpg',0)
# noise_img = util.random_noise(img,mode='poisson')
# noise_img = util.random_noise(img,mode='s&p')
# noise_img = util.random_noise(img,mode='speckle')
noise_img = util.random_noise(img,mode='gaussian')
# print(noise_img)
# 将浮点数类型的噪声图像转换为8位无符号整数类型，并限制值的范围 需要图像拼接时候使用
noise_img = (np.clip(noise_img, 0, 1) * 255).astype(np.uint8)
cv2.imshow('src',img)
cv2.imshow('noise',noise_img)
# 水平拼接原始图像和噪声图像
cv2.imshow('src_noise',np.hstack([img,noise_img]))
cv2.waitKey()
# print(noise_img)

