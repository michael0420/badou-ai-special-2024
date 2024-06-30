import cv2
import numpy as np


def function(img,h,w):
    height,width,channels = img.shape
    EmptyImage = np.zeros((h,w,channels),np.uint8)
    sh = h/height
    sw = w/width
    for i in range(h):
        for j in range(w):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            EmptyImage[i,j] = img[x,y]
    return EmptyImage

img = cv2.imread('lenna.png')

zoom = function(img,800,800)
print(zoom)
print(zoom.shape)

cv2.imshow('nearest',zoom)
cv2.imshow('image',img)
cv2.waitKey(0)
