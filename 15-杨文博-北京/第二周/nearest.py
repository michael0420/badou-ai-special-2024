import cv2
import numpy as np
def function(img):
    h,w,channels =img.shape
    newimage=np.zeros((800,800,channels),np.uint8)
    sh = 800 / h
    sw = 800 / w
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5)      #四舍五入
            y=int(j/sw + 0.5)
            newimage[i,j]=img[x,y]
    return newimage

img=cv2.imread("lenna.png")
big=function(img)
cv2.imshow("image",img)
cv2.imshow("nearest",big)
print(big)
cv2.waitKey(0)
