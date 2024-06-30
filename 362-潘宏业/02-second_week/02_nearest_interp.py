import cv2
import numpy as np


def function(img, size):
    h, w, c = img.shape
    new_image = np.zeros((size[0], size[1], c), np.uint8)
    sh = size[0]/h
    sw = size[1]/w
    for i in range(size[0]):
        for j in range(size[1]):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            new_image[i, j] = img[x, y]
    return new_image




img = cv2.imread("lenna.png")
h = int(input("图片的高"))
w = int(input("图片的宽"))
size = [h, w]
zoom = function(img, size)
cv2.imshow("initial", img)
cv2.imshow("enlarge", zoom)
cv2.waitKey()