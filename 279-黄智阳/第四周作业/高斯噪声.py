import cv2
import numpy as np
import random
from numpy import shape

def GaussianNoise(src,means,sigma,percetage):
    Noiseimg=src
    Noisenum=int(percetage*Noiseimg.shape[0]*Noiseimg.shape[1])
    for i in range(Noisenum):
        randX=random.randint(0,Noiseimg.shape[0]-1)
        randY=random.randint(0,Noiseimg.shape[1]-1)
        Noiseimg[randX][randY]=src[randX][randY]+random.gauss(means,sigma)
        if Noiseimg[randX][randY]<0:
            Noiseimg[randX][randY]=0
        elif Noiseimg[randX][randY]>255:
            Noiseimg[randX][randY]=255
    return Noiseimg

img=cv2.imread('lenna.png',0)
Gausimg=GaussianNoise(img,2,4,0.8)
img=cv2.imread('lenna.png')
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("grayimg",grayimg)
cv2.imshow("Gaussimg",Gausimg)
cv2.waitKey(0)
