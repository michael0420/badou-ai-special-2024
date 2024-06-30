import numpy as np
import cv2
import random

def add_noise(src,perc):
    noise_img = src
    noise_num = int(src.shape[0] * src.shape[1] * perc)

    for i in range(noise_num):
        randX = random.randint(0,src.shape[0] - 1)
        randY = random.randint(0,src.shape[1] - 1)

        if random.random() <= 0.5:
            noise_img[randX,randY] = 0
        else:
            noise_img[randX,randY] = 255
    return noise_img

img = cv2.imread('./lenna.png',0)
img1 = add_noise(img,0.1)
cv2.imwrite('lenna_pepper_add.png',img1)

img = cv2.imread('lenna.png',0)
img2 = cv2.imread('lenna_pepper_add.png',0)

cv2.imshow('src',img)
cv2.imshow('add_pepper',img2)
cv2.waitKey(0)