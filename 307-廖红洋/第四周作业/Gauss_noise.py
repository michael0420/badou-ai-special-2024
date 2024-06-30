# -*- coding: utf-8 -*-
'''
@File    :   Gauss.py
@Time    :   2024/04/18 16:11:13
@Author  :   廖红洋 
'''
import cv2
import random

img = cv2.imread("lenna.png",0)
means = 8 # 噪声平均值
sigma = 6 # 标准方差，即噪声分布均匀程度，越大越接近椒盐噪声
percent = 0.6 # 加噪声百分比
num = int(img.shape[0]*img.shape[1]*percent)
for i in range(num):
    rdx = random.randint(0,img.shape[0]-1)
    rdy = random.randint(0,img.shape[1]-1)
    img[rdx,rdy] = img[rdx,rdy] + random.gauss(means,sigma)
    if  img[rdx,rdy]< 0: # 修正超过unit8表示范围的值
        img[rdx,rdy]=0
    elif img[rdx,rdy]>255:
        img[rdx,rdy]=255
cv2.imshow("gray",img)
cv2.imwrite("lenna_gauss.png",img)
cv2.waitKey(0)