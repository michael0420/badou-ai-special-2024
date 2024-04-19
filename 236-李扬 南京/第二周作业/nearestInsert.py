import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

x = cv.imread("lenna.png")
cv.imshow("test", x)
w, h, channel = x.shape[:3]
wradio = w/800
hradio = h/800
newImg = np.zeros((800, 800, channel), np.uint8)

for i in range(800):
    for j in range(800):
        xi = int(i * wradio + 0.5)
        yi = int(j * hradio + 0.5)
        newImg[i, j] = x[xi, yi]

cv.imshow("test1", newImg)
cv.waitKey()