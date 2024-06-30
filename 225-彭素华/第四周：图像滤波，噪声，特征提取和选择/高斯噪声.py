import numpy as np
import cv2
from numpy import shape
import random


def GaussianNoise(src, means, sigma, percentage):
    NoiseImg = src
    NOiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NOiseNum):
        '''
        每次取一个随机点，把每一张图像对 像素用行和列表示的话，randX代表随机生成的行，randY代表随机生成的列。
        random.randint()生成随机整数
        高斯噪声边缘不处理，故：-1
        '''
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        '''
        此处在原有像素灰度值上加上随机数
        '''
        NoiseImg[randY, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        '''
        若灰度值小于0则强制为0，若灰度值大于255则强制为255.
        '''
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)
cv2.imshow("lenna-GuessNOise", img1)
cv2.waitKey(0)
