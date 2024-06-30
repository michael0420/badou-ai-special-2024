#!/usr/bin/env python
# coding: utf-8


"""
import cv2
import numpy as np
from numpy import shape
import random


def pepper_noise (img, percentage):
    #读取图片信息
    pepper_img = np.copy(img)
    height = pepper_img.shape[0]
    width = pepper_img.shape[1]
    #计算添加噪声的数量和位置
    num = int(percentage*height*width)
    for i in range(num):
        random_x = random.randint(0, height - 1)
        random_y = random.randint(0, width - 1)
        rand = random.random()  
        if rand <= 0.5:
            pepper_img[random_x, random_y] = 0
        else:
            pepper_img[random_x, random_y] = 255
    return pepper_img

"""
注意缩进return的位置
"""

img = cv2.imread('lenna.png')
pepper_img = pepper_noise(img, 0.1)
cv2.imshow("pepperimg", pepper_img)

cv2.waitKey(0)

