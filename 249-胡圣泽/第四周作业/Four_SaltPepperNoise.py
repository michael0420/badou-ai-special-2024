import cv2
import numpy as np
from numpy import shape
import random


# 高斯噪声函数，随机点增加random.gauss噪声
def SaltPepperNoise(img, per):
    NewImg = img
    NNum = int(per * img.shape[0] * img.shape[1])
    for i in range(NNum):
        randomX = random.randint(0, img.shape[0] - 1)
        randomY = random.randint(0, img.shape[1] - 1)
        # 万一灰度值超出范围，则置于极值
        if random.random() <= 0.5:
            NewImg[randomX, randomY] = 0
        else:
            NewImg[randomX, randomY] = 255
    return NewImg


img = cv2.imread('lenna.png', 0)
newimg = SaltPepperNoise(img, 0.5)
img2 = cv2.imread('lenna.png', 0)
cv2.imshow('SourceImg', img2)
cv2.imshow('SaltPepperNoiseImg', newimg)
cv2.waitKey(0)