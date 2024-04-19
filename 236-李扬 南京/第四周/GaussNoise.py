import numpy
import cv2 as cv
import random
import numpy as np
from skimage import util

def gaussnoise(sigma, mean, img, precent):
    h, w= img.shape[:]
    # newImg = np.zeros((h, w), dtype=np.uint8)
    newImg = img
    noiseNum = int(precent * h * w)
    for i in range(noiseNum):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        newImg[y, x] = img[y, x] + random.gauss(mean, sigma)
        if newImg[y, x] < 0:
            newImg[y, x] = 0
        elif newImg[y, x] > 255:
            newImg[y, x] = 255
    return newImg

img = cv.imread("lenna.png", 0)
newImg1 = gaussnoise(5, 3, img, 0.8)
cv.imshow('newImg', newImg1)
img1 = cv.imread("lenna.png", 0)
cv.imshow('Img', img1)
#调用接口实现
noise_gs_img = util.random_noise(img, mode='gaussian')
cv.imshow('Img1', noise_gs_img)
cv.waitKey()


