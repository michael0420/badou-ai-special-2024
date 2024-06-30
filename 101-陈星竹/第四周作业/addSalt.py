import cv2
import numpy as np
import random

def add_salt(img,percent):
    h,w = img.shape
    saltImg = img
    noisy_count = int(percent*h*w)
    for i in range(noisy_count):
        randomX = random.randint(0,h-1)
        randomY = random.randint(0,w-1)
        saltImg[randomX,randomY] = random.choice([0,255])
    return saltImg

img = cv2.imread('lenna.png',0)
saltImg = add_salt(img,0.5)

cv2.imshow('salt',saltImg)
cv2.waitKey(0)