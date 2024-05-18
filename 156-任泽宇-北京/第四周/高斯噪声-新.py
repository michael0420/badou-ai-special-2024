import cv2
import random


def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseSum = int(src.shape[0] * src.shape[1] * percetage)

    for i in range(NoiseSum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)

        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)
img2 = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(0)
