import cv2
import numpy as np

def function(img):
    h,w,c=img.shape
    Emptyimg=np.zeros((800,800,c),np.uint8)
    sh=800/h
    sw=800/w
    for i in range(800):
        for j in range(800):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
            Emptyimg[i,j]=img[x,y]

    return Emptyimg
    # cv2.resize(img, (800,800,c),near/bin)

img=cv2.imread("lenna.png")
zoom=function(img)
print("zoom-----",zoom)
print("zoomshape=====",zoom.shape)
cv2.imshow("dasdadad",zoom)
cv2.waitKey()


