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

def fun(img):
    height,width,channals=img.shape[:]            #原图是512*512，取出长，宽，通道数
    newimg=np.zeros((1000,1000,channals),np.uint8)      #创建一个大小为1000*1000大小的图像
    sh=1000/height             #求放大比例
    sw=1000/width
    for i in range(1000):          #列出1000*1000目标图像上每个像素点
        for j in range(1000):
            x=int(i/sh+0.5)         #int(),转为整型，向下取整:int(3.6)=3,所以3.6+0.5>=4
            y=int(j/sw + 0.5)
            newimg[i,j]=img[x,y]
    return newimg
Img=cv2.imread("lenna.png")
newImg=fun(Img)
cv2.imshow("lenna.png",Img)
cv2.imshow("newImg.png",newImg)
cv2.waitKey(0)



