import cv2
import numpy as np

img = cv2.imread("lenna.png")
cv2.imshow('lenna', img)
h, w, c = img.shape
empty_img = np.zeros([800, 800, c], img.dtype)
print(empty_img)
for i in range(800):
    for j in range(800):
        x = int(i/800*h+0.5)
        y = int(j/800*w+0.5)
        empty_img[i, j] = img[x, y]
print(empty_img)
cv2.imshow('changed_img', empty_img)
cv2.waitKey(0)
