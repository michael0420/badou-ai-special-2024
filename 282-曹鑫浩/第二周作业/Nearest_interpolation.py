import cv2
import numpy as np

img_original = cv2.imread("lenna.png", 1)
h, w = img_original.shape[:2]
img_dst = np.zeros([800, 800, 3], img_original.dtype)

for i in range(800):
    for j in range(800):
        img_dst[i, j] = img_original[int(i*h/800 + 0.5), int(j*w/800 + 0.5)]

cv2.imshow('img_dst', img_dst)
cv2.imshow('img_original', img_original)
key = cv2.waitKey(5000)
cv2.destroyAllWindows()

