#!/usr/bin/env python
# coding: utf-8

# In[24]:


import cv2 as cv
from skimage import util
import numpy as np
from sklearn.decomposition import PCA
img = cv.imread('lenna.png')

# 整数转换成浮点数
# img_float = img.astype(np.float32) / 255.0 

# 添加高斯噪声
noise_gs_img0=util.random_noise(img,mode='gaussian')

# 添加椒盐噪声
noise_gs_img1=util.random_noise(img,mode='s&p')

# 浮点数转换成整数
# G= (noise_gs_img0*255).astype(np.uint8) 
# J= (noise_gs_img1*255).astype(np.uint8) 

# 图像灰度化处理
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# PCA降维处理
pca=PCA(n_components=350)                 
newimg=pca.fit_transform(gray) 

# 打印属性
print(img.shape)
print(noise_gs_img0.shape)
print(noise_gs_img1.shape)
print(gray.shape)
print(newimg.shape)
h, w = img.shape[:2]  
print("图片高度:", h)
# cv.imshow('source',newimg)

# 水平显示图片
cv.imshow("zaosheng",np.hstack([img/255,noise_gs_img0,noise_gs_img1]))
cv.waitKey(0)
cv.destroyAllWindows()

