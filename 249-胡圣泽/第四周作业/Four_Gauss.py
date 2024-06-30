import cv2
import numpy as np
from numpy import shape
import random


# 高斯噪声函数，随机点增加random.gauss噪声
def GaussNoise(img, means, sigma, per):
    NewImg = img
    NNum = int(per * img.shape[0] * img.shape[1])
    for i in range(NNum):
        randomX = random.randint(0, img.shape[0] - 1)
        randomY = random.randint(0, img.shape[1] - 1)
        NewImg[randomX, randomY] = NewImg[randomX, randomY] + random.gauss(means, sigma)
        # 万一灰度值超出范围，则置于极值
        if NewImg[randomX, randomY] < 0:
            NewImg[randomX, randomY] = 0
        elif NewImg[randomX, randomY] > 255:
            NewImg[randomX, randomY] = 255
    return NewImg


img = cv2.imread('lenna.png', 0)
newimg = GaussNoise(img, 2, 4, 1)
img2 = cv2.imread('lenna.png', 0)
cv2.imshow('SourceImg', img2)
cv2.imshow('SaltPepperNoiseImg', newimg)
cv2.waitKey(0)