import numpy as np
import cv2
from numpy import shape
import random

def perpper_noise(src,percetage):
    pepnoise = src
    pepnum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(pepnum):
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        if random.random() < 0.5:
            pepnoise[randx,randy] = 0
        else:
            pepnoise[randx,randy] = 255
    return pepnoise
img = cv2.imread("lenna.png",0)
img1 = perpper_noise(img,0.2)
cv2.imshow('pepper',img1)
cv2.waitKey()
