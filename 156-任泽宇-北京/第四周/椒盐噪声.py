import cv2
import random


def jiaoyan(src, percetage):
    NoiseImg = src
    NoiseSum = int(src.shape[0] * src.shape[1] * percetage)

    for i in range(NoiseSum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)

        if random.random() < 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png')
img1 = jiaoyan(img, 0.1)
img2 = cv2.imread('lenna.png')
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(10000)