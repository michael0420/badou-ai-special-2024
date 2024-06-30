import cv2
import numpy as np
import random
from skimage import util


def add_gauss_noise(img, mean=2, sigma=4, percentage=0.6):
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


img_gray = cv2.imread("img/lenna.png", 2)
img_gauss = add_gauss_noise(img_gray, 2, 4, 1)

cv2.imshow("Gray Image", img_gray)
cv2.imshow("Gaussian Noise Image", img_gauss)
cv2.waitKey(0)
