import numpy as np
import cv2
import random


def GaussianNoise(src, means, sigma, percentage):
    NoiseImage = src.copy()
    NoiseNum = int(percentage * NoiseImage.shape[0] * NoiseImage.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0]-1)  # random.randint生成随机整数
        randY = random.randint(0, src.shape[1]-1)  # 高斯噪声图片边缘不处理，故-1
        # 此处在原有像素灰度值上加上服从高斯分布的随机数
        NoiseImage[randX, randY] = NoiseImage[randX, randY] + random.gauss(means, sigma)
        # 灰度值检测
        if NoiseImage[randX, randY] < 0:
            NoiseImage[randX, randY] = 0
        elif NoiseImage[randX, randY] > 255:
            NoiseImage[randX, randY] = 255
    return NoiseImage


img = cv2.imread("lenna.png", 0)  # 读取灰度图像
img1 = GaussianNoise(img, 5, 2, 1)
cv2.imshow('source', img)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(0)










