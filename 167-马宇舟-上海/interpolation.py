import cv2
import numpy as np
def function(img):
    height, width, channels = img.shape
    empty_image = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            empty_image[i, j] = img[x, y]
    return empty_image

img = cv2.imread(r"C:\Users\lenovo\Desktop\lenna.png")
zoom = function(img)
cv2.imshow('hello world', zoom)
cv2.imshow('initial', img)
cv2.waitKey(0)
