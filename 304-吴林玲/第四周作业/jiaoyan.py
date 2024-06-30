import numpy as np
import cv2
from numpy import shape
import random

#实现椒盐噪声
def jiaoyan(src,percentage):
    NoiseImg = src
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.random() <= 0.5:
            NoiseImg[randX,randY] = 0
        else:
            NoiseImg[randX,randY] = 255
    return NoiseImg


if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)
    img1 = jiaoyan(img, 0.5)
    img = cv2.imread('lenna.png')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source', img2)
    cv2.imshow('lenna_jiaoyan', img1)
    cv2.waitKey(0)

