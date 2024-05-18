import numpy as np
import cv2
from numpy import shape
import random


def  PepperAndSalt(rawImg,percetage):
    NoiseImg=rawImg
    NoiseNum=int(percetage*rawImg.shape[0]*rawImg.shape[1])
    for i in range(NoiseNum):
	    randX=random.randint(0,rawImg.shape[0]-1)   # 获得0到原图横轴-1的随机数
	    randY=random.randint(0,rawImg.shape[1]-1)   # 获得0到原图纵轴-1的随机数
	    #random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0
	    if random.random()<=0.5:
	    	NoiseImg[randX,randY]=0
	    else:
	    	NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread('lenna.png',0)
img1=PepperAndSalt(img,0.3)

cv2.imwrite('lenna_PepperandSalt.png',img1) #在指定的文件夹写入加椒盐噪声后的图片

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)

