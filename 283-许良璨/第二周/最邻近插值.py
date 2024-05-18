import cv2
import numpy as np

def alterimge(img):
    height,width,channels=img.shape
    emptyimage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for g in range(800):
            x=int(i/sw+0.5)
            y=int(g/sh+0.5)
            emptyimage[i,g]=img[x,y]
    return emptyimage

img=cv2.imread("lenna.png")
zoom = alterimge(img)
cv2.imshow("完成",zoom)