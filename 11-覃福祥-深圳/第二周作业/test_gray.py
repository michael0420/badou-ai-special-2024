# -*- coding: utf-8 -*-
"""
@author: Michael

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img = plt.imread("lenna.png")
plt.subplot(221)
plt.title("原图")
plt.xticks([]), plt.yticks([]) # 隐藏x和y轴
plt.imshow(img)

# 灰度化
img = cv2.imread("lenna.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]                               #获取图片的high和wide
img_gray=np.zeros([h,w],img.dtype)                #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j]                              #取出当前high和wide中的BGR坐标
        img_gray[i,j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)   #将BGR坐标转化为gray坐标并赋值给新图像
#img_gray=rgb2gray(img)                                         #直接调用 rgb2gray库进行灰度化
# img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                  #直接调用 rgb2gray库进行灰度化
#cv2.imshow("image show gray",img_gray)
plt.subplot(222)
plt.title("灰度图")
plt.xticks([]), plt.yticks([]) # 隐藏x和y轴
plt.imshow(img_gray, cmap='gray')


#二值化
def img_binaryzation(m):       #m是二值化阈值
    r,c=img_gray.shape[:]
    img_binary=np.zeros([r,c],img_gray.dtype)
    for i in range(r):
        for j in range(c):
            if(img_gray[i,j]<=m):
                img_binary[i,j]=0
            else:
                img_binary[i,j]=1
    return img_binary
img_binary=img_binaryzation(128)

#img_binary = np.where(img >= 125, 1, 0)
plt.subplot(223)
plt.title("二值化图")
plt.xticks([]), plt.yticks([]) # 隐藏x和y轴
plt.imshow(img_binary, cmap='gray')



plt.show()
cv2.waitKey(0)