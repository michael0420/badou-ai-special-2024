import cv2 as cv
import random
import numpy as np

img = cv.imread('../week2/lenna.png', 0)
h, w = img.shape[:2]
result = np.zeros(img.shape, dtype=np.uint8)
for i in range(h):
    for j in range(w):
        tmp = img[i, j] + random.gauss(0, 16)
        if tmp > 255:
            result[i, j] = 255
        elif tmp < 0:
            result[i, j] = 0
        else:
            result[i, j] = tmp

img_color = cv.imread('../week2/lenna.png', 1)
row, col, channel = img_color.shape
result_color = np.zeros(img_color.shape, dtype=np.uint8)
for c in range(channel):
    for i in range(row):
        for j in range(col):
            tmp = img_color[i, j, c] + random.gauss(0, 25)
            if tmp > 255:
                result_color[i, j, c] = 255
            elif tmp < 0:
                result_color[i, j, c] = 0
            else:
                result_color[i, j, c] = tmp

cv.imshow('color-gauss-copmpare', np.hstack([img_color, result_color]))
cv.imshow('gray-gauss-copmpare', np.hstack([img, result]))
cv.waitKey(0)
