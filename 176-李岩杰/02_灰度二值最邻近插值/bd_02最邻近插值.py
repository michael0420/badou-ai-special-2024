import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape
    empty_img = np.zeros((800, 800, 3), np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            empty_img[i, j] = img[x, y]
    return empty_img


img = cv2.imread("lenna.png")
zoom = function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
