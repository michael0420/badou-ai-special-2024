import numpy as np
import cv2
from numpy import shape
import random

#实现高斯噪声
def GaussianNoise(src,means,sigma,percetage):
    NoiseImg = src
    #计算需要处理的像素点
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    #利用随机数选取随意的坐标进行像素点的值处理
    for i in range(NoiseNum):
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)
        NoiseImg[randX,randY] = NoiseImg[randX,randY] + random.gauss(means,sigma)
        if NoiseImg[randX,randY] < 0:
            NoiseImg[randX,randY] = 0
        elif NoiseImg[randX,randY] > 255:
            NoiseImg[randX,randY] = 255

    return NoiseImg

if __name__ == '__main__':
    img = cv2.imread('lenna.png', 0)
    img1 = GaussianNoise(img, 2, 4, 0.8)
    img = cv2.imread('lenna.png')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('source', img2)
    cv2.imshow('lenna_GaussianNoise', img1)
    cv2.waitKey(0)
