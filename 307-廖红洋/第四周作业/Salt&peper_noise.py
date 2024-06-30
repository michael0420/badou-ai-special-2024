# -*- coding: utf-8 -*-
'''
@File    :   Salt&peper_noise.py
@Time    :   2024/04/18 23:22:40
@Author  :   廖红洋 
'''

import cv2
import random

img = cv2.imread("lenna.png",0)
percent = 0.02 # 加噪声百分比
num = int(img.shape[0]*img.shape[1]*percent)
for i in range(num):
    rdx = random.randint(0,img.shape[0]-1)
    rdy = random.randint(0,img.shape[1]-1)
    if random.random()>0.5: #直接调用random生成0-1之间的随机数
        img[rdx,rdy] = 255
    else:
        img[rdx,rdy] = 0
cv2.imshow("gray",img)
cv2.imwrite("lenna_salt.png",img)
cv2.waitKey(0)
