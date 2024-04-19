import cv2
import numpy as np
def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800,800,channels), np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5) #int(), 转为整型，使用向下取整
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread('lenna.png')
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow('nearest interp',zoom)
cv2.imshow('image',img)
cv2.waitKey(0)