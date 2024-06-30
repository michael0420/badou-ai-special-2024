import numpy as np
import cv2
from numpy import shape 
import random

def GaussianNoise(src,means,sigma,percetage):
    GauImg = src
    # 获取需要处理的像素点个数
    GauNum = int(src.shape[0] * src.shape[1] * percetage)
    for i in range(GauNum):
        # 随机取x,y
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)
        # 加高斯噪声
        GauImg[randX, randY] += random.gauss(means, sigma)
        # 去除非法数
        if GauImg[randX, randY] < 0:
            GauImg[randX, randY] = 0
        if GauImg[randX, randY] > 255:
            GauImg[randX, randY] = 255
    return GauImg

img = cv2.imread('test.png',0)
img1 = GaussianNoise(img, 2, 4, 0.8)
img = cv2.imread('test.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('test_GaussianNoise',img1)
cv2.waitKey(0)