import numpy as np
import cv2
from skimage import util
import random

def gauss_noise(img, mean=2, sigma=4, percentage=0.6):
    img_copy = img.copy()
    height,width = img.shape[:2]
    number = int(width * height * percentage)
    for i in range(number):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        img_copy[x, y] = img_copy[x, y] + random.gauss(mean, sigma)
        if img_copy[x, y] > 255:
            img_copy[x, y] = 255
        elif img_copy[x, y] < 0:
            img_copy[x, y] = 0

    return img_copy

original_img = cv2.imread("lenna.png", 2)

gauss_noise_img = gauss_noise(original_img, 2, 4, 1)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gs_noise_img = util.random_noise(img, mode='gaussian')

cv2.imshow('original_img', original_img)
cv2.imshow("gauss_noise_img", gauss_noise_img)
cv2.waitKey(0)