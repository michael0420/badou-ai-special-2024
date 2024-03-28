import cv2
import numpy as np

def function(img):
    height,width,channels = img.shape
    empty_image = np.zeros((1000,1000,channels),np.uint8)
    sh = 1000/height
    sw = 1000/width
    for i in range(1000):
        for j in range(1000):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            empty_image[i,j] = img[x,y]
    return empty_image

# 图片显示
img = cv2.imread('Grace.jpg')
zoom = function(img)
cv2.imshow('nearest interp',zoom)
cv2.imshow('original',img)
cv2.waitKey(0)
