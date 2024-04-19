import cv2
import numpy as np


def function(img):
    h, w, channels = img.shape
    newImage = np.zeros((800, 800, channels), img.dtype)
    sh = 800 / h
    sw = 800 / w
    for i in range(800):
        for j in range(800):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            newImage[i, j] = img[x, y]
    return newImage


img = cv2.imread("lenna.png")
newImage = function(img)
print("image", img)
print("nearest interpolation", newImage)
cv2.imshow("nearest interpolation", newImage)
cv2.imshow("image", img)
cv2.waitKey(0)
