#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
# 读取图像
img =plt.imread('lenna.png')
# 将原来的彩图灰度化
img_gray = rgb2gray(img)
# 将灰度化的图形像素取临界值，大于等于0.5 取1，小于0.5取0
img_binary =np.where(img_gray >=0.5, 1, 0)
# 显示处理后的二值化图像
plt.imshow(img_binary, cmap = "gray")

