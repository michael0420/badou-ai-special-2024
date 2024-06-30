import numpy as np
import cv2
import random

def pepper_salt_noise(img, persentage=0.3):
    img_copy = img.copy()
    height, width = img_copy.shape[:2]
    number = int(persentage * height * width)
    for i in range(number):
        x = random.randint(1, width - 1)
        y = random.randint(1, height - 1)
        if random.random() < 0.5:
            img_copy[y, x] = 0
        else:
            img_copy[y, x] = 255
    return img_copy

original_img = cv2.imread("lenna.png", 2)

cv2.imshow('original_img',original_img)
pepper_salt_noise_img = pepper_salt_noise(original_img, 0.8)
cv2.imshow('pepper_salt_noise_img', pepper_salt_noise_img)
cv2.waitKey(0)