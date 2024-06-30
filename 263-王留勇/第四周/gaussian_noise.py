"""
高斯噪声
"""

import cv2
import random

def gaussian_noise(src, means, sigma, percetage):
    noiseImg = src
    height, width = noiseImg.shape[:2]
    noiseNum = int(percetage * height * width)
    for i in range(noiseNum):
        randX = random.randint(0, height - 1)
        randY = random.randint(0, width - 1)
        # 在原像素灰度值上加上随机数
        noiseImg[randX, randY] = noiseImg[randX, randY] + random.gauss(means, sigma)
        # 控制在 0~255范围
        if noiseImg[randX, randY] < 0:
            noiseImg[randX, randY] = 0
        elif noiseImg[randX, randY] > 255:
            noiseImg[randX, randY] = 255
    return noiseImg


img = cv2.imread('lenna.png', 0)
gaussianImg = gaussian_noise(img, 2, 4, 0.8)
cv2.imshow('source', img)
cv2.imshow('gaussian img', gaussianImg)
cv2.waitKey(0)