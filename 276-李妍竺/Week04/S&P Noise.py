import cv2
import numpy as np
from numpy import shape
import random

def P_S(src,percentage):
    Noise_img = src
    Noise_num = int(percentage * src.shape[0] * src.shape[1])
    for i in range(Noise_num):
        rand_x = random.randint(0, src.shape[0] - 1)
        rand_y = random.randint(0, src.shape[1] - 1)
        #random.random()：生成随机浮点数，[0,1]
        if random.random()<=0.5:
            Noise_img[rand_x , rand_y] = 0
        else:
            Noise_img[rand_x , rand_y] = 255
    return Noise_img

img = cv2.imread('lenna.png',0)
img1 = P_S(img,0.5)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow('source',img2)
cv2.imshow('Pepper_Salt',img1)
cv2.waitKey()
