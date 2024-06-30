import numpy as np
import cv2
from numpy import shape 
import random

def func(src,percetage):
    GauImg = src
    # 获取需要处理的像素点个数
    GauNum = int(src.shape[0] * src.shape[1] * percetage)
    for i in range(GauNum):
        # 随机取x,y
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)
        # 加椒盐噪声
        rand = random.random()
        if rand < 0.5:
            GauImg[randX, randY] = 0
        else:
           GauImg[randX, randY] = 255
    return GauImg

img = cv2.imread('test.png',0)
img1 = func(img, 0.2)
img = cv2.imread('test.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('test_Noise',img1)
cv2.waitKey(0)