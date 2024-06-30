import numpy as np
import cv2
from numpy import shape
import random


def fun1(src, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        if random.random() <= 0.5:
            NoiseImg[randx, randy] = 0
        else:
            NoiseImg[randx, randy] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = fun1(img, 0.05)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('yuan', img2)
cv2.imshow('lenna_PepperandSalt', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

