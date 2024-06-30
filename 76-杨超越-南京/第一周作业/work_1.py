# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:33:08 2024
作业：
@author: DELL
"""

import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

lenna_brg = cv2.imread('lenna.png', -1)
h, w = lenna_brg.shape[:2]
# 用plt库画图，需要将
# lenna = lenna_brg[:, :, ::-1]
lenna = lenna_brg.copy()
lenna[:, :, 0] = lenna_brg[:, :, 2]
lenna[:, :, 2] = lenna_brg[:, :, 0]
# lenna[:, :, 2] = lenna_brg[:, :, 2]

lenna_gray = np.zeros([h, w], lenna.dtype)

plt.subplot(1, 3, 1)
plt.xticks([]), plt.yticks([])      #隐藏x轴和y轴
plt.imshow(lenna)

# 灰度化照片
for i in range(h):               # 将RGB图片变化为gray图片
    for j in range(w):
        lenna_gray[i, j] = int(lenna[i, j, 0]*0.114 + lenna[i, j, 1]*0.587 + lenna[i, j, 2]*0.299)
        
plt.subplot(1, 3, 2)
plt.xticks([]), plt.yticks([])      #隐藏x轴和y轴
plt.imshow(lenna_gray, cmap='gray')
# 实现二值图
lenna_binary = np.where(lenna_gray >= 112.5, 1, 0) 
plt.subplot(1, 3, 3)
plt.xticks([]), plt.yticks([])      #隐藏x轴和y轴
plt.imshow(lenna_binary, cmap='gray')
























