import numpy as np
import matplotlib.pyplot as plt
import cv2


def nearest_interp(img, newHeight, newWidth):
    height, width = img.shape[:2]
    dx = width / newWidth
    dy = height / newHeight
    newImg = np.zeros((newHeight, newWidth, 3), dtype=np.uint8)
    for r in range(newHeight):
        for c in range(newWidth):
            newImg[r, c] = img[int(r * dy + 0.5), int(c * dx + 0.5)]

    return newImg


img = cv2.imread('img/lenna.png')
newImg = nearest_interp(img, 800, 800)
cv2.imshow('original', img)
cv2.imshow('nearest_interp', newImg)
cv2.waitKey(0)
