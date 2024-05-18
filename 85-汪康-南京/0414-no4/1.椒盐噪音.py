# -*- coding: utf-8 -*-
'''@Time: 2024/4/14 11:05

'''
import random
import numpy as np
import cv2

def fun1(src,percetage):
    NoiseImg = src
    h,w = src.shape[0],src.shape[1]
    NoiseNum = int(percetage*h*w)
    for i in range(NoiseNum):
        randX = random.randint(0,h-1)
        randY = random.randint(0,w-1)
        if random.random()<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg
img = cv2.imread("../lenna.png",0)
img1=fun1(img,0.2)
img = cv2.imread("../lenna.png")
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('source Noise',np.hstack([img2,img1]))
cv2.waitKey(0)

