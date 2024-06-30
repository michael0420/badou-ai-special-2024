"""
@author: Hanley-Yang

椒盐噪声
"""

import cv2
import random
def saltpepperNoise(src, percentage):
    noiseImg = src
    noiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(noiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.random() <= 0.5:
            noiseImg[randX, randY] = 0
        else:
            noiseImg[randX, randY] = 255
    return noiseImg

srcImg = cv2.imread('shangri-la.jpg',0)
saltpepperImg = saltpepperNoise(srcImg, 0.8)

cv2.imwrite('saltpepper_noise.jpg', saltpepperImg)

srcImg = cv2.imread('shangri-la.jpg')
srcImg_gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)

cv2.imshow('source', srcImg_gray)
cv2.imshow('PepperandSaltNoise',saltpepperImg)

cv2.waitKey(0)
