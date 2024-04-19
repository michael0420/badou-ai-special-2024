import cv2
import numpy as np


def function(img):
    high, wide, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    xh = 800 / high
    xw = 800 / wide
    for i in range(800):
        for j in range(800):
            x = int(i / xh + 0.5)
            y = int(j / xw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread('lenna.png')
img1 = function(img)
cv2.imshow('new', img1)
cv2.imshow('image', img)
cv2.waitKey(0)
