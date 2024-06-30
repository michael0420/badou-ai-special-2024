import cv2
import numpy as np
import random

def add_gauss(img,mean,stu,precent):
    h,w = img.shape
    noisy_count = int(h*w*precent)
    noisyImg = img
    for i in range(noisy_count):
        randomX = random.randint(0,h-1)
        randomY = random.randint(0,w-1)

        noisyImg[randomX,randomY] = noisyImg[randomX,randomY] + random.gauss(mean,stu)
        if noisyImg[randomX,randomY] < 0 :
            noisyImg[randomX,randomY] = 0
        if noisyImg[randomX,randomY] > 255 :
            noisyImg[randomX,randomY] = 255
    return noisyImg

img = cv2.imread('lenna.png',0)
gauss_img = add_gauss(img,5,2,1)

cv2.imshow('gauss',gauss_img)
cv2.waitKey(0)