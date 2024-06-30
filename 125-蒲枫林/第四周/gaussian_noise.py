import numpy as np
import random
import cv2

def gaussian_noise(src,perc,mean,sigma):
    noise_img = src
    noise_num = int(src.shape[0] * src.shape[1] * perc)
    for i in range(noise_num):
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)

        noise_img[randX,randY] = noise_img[randX,randY] + random.gauss(mean,sigma)

        if noise_img[randX,randY] < 0:
            noise_img[randX,randY] = 0
        elif noise_img[randX,randY] > 255:
            noise_img[randX,randY] = 255

    return noise_img

src = cv2.imread('lenna.png',0)
img1 = gaussian_noise(src,0.1,20,40)

cv2.imwrite('lenna_gaussian_add.png',img1)

img = cv2.imread('lenna.png',0)

cv2.imshow('src',img)
cv2.imshow('lenna_gaussian_noise',img1)
cv2.waitKey(0)
