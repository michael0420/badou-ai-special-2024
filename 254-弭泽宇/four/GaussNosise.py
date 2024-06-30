import cv2
import numpy as np
import random

def GaussNoise(src,means,sigma,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):

      randX=random.randint(0,src.shape[0]-1)
      randY=random.randint(0,src.shape[1]-1)

      NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)

      if NoiseImg[randX,randY]<0:
          NoiseImg[randX,randY]=0

      if NoiseImg[randX,randY]>0:
          NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread("lenna.png",0)
img1=GaussNoise(img,2,5,0.4)
img2=cv2.imread("lenna.png")
cv2.imshow("img",img2)
cv2.imshow("img1",img1)
cv2.waitKey()