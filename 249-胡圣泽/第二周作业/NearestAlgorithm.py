import cv2
import numpy as np


#最邻近算法
def Algorithm(img):
    chicun = 1000
    height,width,channels = img.shape
    NewImage = np.zeros((chicun,chicun,channels),np.uint8)
    sh = chicun/height
    sw = chicun/width
    for i in range(chicun):
        for j in range(chicun):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            #x = int(i/sh)
            #y = int(j/sw)
            NewImage[i,j]=img[x,y]
    return NewImage


img = cv2.imread("lenna.png")
newimg = Algorithm(img)
print(newimg)
print(newimg.shape)
cv2.imshow("nearest algorithm",newimg)
cv2.imshow("image",img)
cv2.waitKey(0)
