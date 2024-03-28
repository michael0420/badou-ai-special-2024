# _*_ coding: UTF-8 _*_
import cv2
import numpy as np


def function(image, height, width):
    h, w, c = image.shape
    new_image = np.zeros((height, width, c), dtype=np.uint8)
    sh = height / h
    sw = width / w
    for i in range(height):
        for j in range(width):
            x = round(i / sh)
            y = round(j / sw)
            new_image[i, j] = img[x, y]
    return new_image


img = cv2.imread('../data/lenna.png')
new_image = function(img, 600, 600)
cv2.imshow('image', img)
cv2.imshow('new image', new_image)
cv2.waitKey(0)
