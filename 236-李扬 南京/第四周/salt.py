import numpy
import cv2 as cv
import random
import numpy as np
from skimage import util

def peppernoise(img, precent):
    h, w= img.shape[:]
    # newImg = np.zeros((h, w), dtype=np.uint8)
    newImg = img
    noiseNum = int(precent * h * w)
    for i in range(noiseNum):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        if random.random() <= 0.5:
            newImg[y, x] = 0
        else:
            newImg[y, x] = 255
    return newImg

img = cv.imread("lenna.png", 0)
newImg1 = peppernoise(img, 0.2)
#保存图片
cv.imwrite('lenna_salt.png', newImg1)
cv.imshow('newImg', newImg1)
img1 = cv.imread("lenna.png", 0)
cv.imshow('Img', img1)
#调用接口实现
noise_salt_img = util.random_noise(img, mode='salt')
cv.imshow('Img1', noise_salt_img)
cv.waitKey()
