
import numpy as np
import cv2
from numpy import shape
import random



def GaussianNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):

        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)

        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        if  NoiseImg[randX, randY]< 0:
            NoiseImg[randX, randY]=0
        elif NoiseImg[randX, randY]>255:
            NoiseImg[randX, randY]=255
    return NoiseImg
img = cv2.imread('NicolasZhaoSi.jpg',0)
img1 = GaussianNoise(img,2,4,0.5)
img = cv2.imread('NicolasZhaoSi.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('NicolasZhaoSi.jpg_GaussianNoise',img1)
cv2.waitKey(0)
