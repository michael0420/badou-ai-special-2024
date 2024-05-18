import numpy as np
import cv2
import random

def gaussian_noise(img, means, sigma, ratio):
    output = img
    NoiseNum = int(img.shape[0] * img.shape[1] * ratio)
    for i in range(NoiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        # 加噪
        output[randX, randY] = output[randX, randY] + random.gauss(means, sigma)
        # 防止数值溢出
        if output[randX, randY] < 0:
            output[randX, randY] = 0
        elif output[randX, randY] > 255:
            output[randX, randY] = 255
        return output

img = cv2.imread('lenna.png',0)
img1 = gaussian_noise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)