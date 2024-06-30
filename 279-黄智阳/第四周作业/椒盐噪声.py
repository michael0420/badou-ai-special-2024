import cv2
import numpy as np
from numpy import shape
import random

def JiaoyanNoise(src,percetage):
    Noiseimg=src
    Noisenum=int(percetage*Noiseimg.shape[0]*Noiseimg.shape[1])
    for i in range(Noisenum):
        randX=random.randint(0,Noiseimg.shape[0]-1)
        randY=random.randint(0,Noiseimg.shape[1]-1)

        if random.random()<0.5:
            Noiseimg[randX][randY]=0
        else:
            Noiseimg[randX][randY]=255
    return Noiseimg

img=cv2.imread('lenna.png',0)
Jiaoyimg = JiaoyanNoise(img,0.8)
img=cv2.imread('lenna.png')
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("grayimg",grayimg)
cv2.imshow("Jiaoyimg",Jiaoyimg)
cv2.waitKey(0)
