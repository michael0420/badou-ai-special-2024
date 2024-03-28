import cv2
import numpy as np
#灰度化
image = cv2.imread('lenna.png')
h,w = image.shape[:2]
image_gray = np.zeros([h,w],image.dtype)
for i in range(h):
    for j in range(w):
        m=image[i,j]
        image_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)

cv2.imshow('image show gray',image_gray)
cv2.waitKey(0)

#二值化
image = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
threshold_value = 128
image_binary = np.zeros_like(image)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        if image[i, j] > threshold_value:
            image_binary[i, j] = 255
        else:
            image_binary[i, j] = 0


cv2.imshow('Manual Binary Image', image_binary)
cv2.waitKey(0)

#最邻近插值
image = cv2.imread('lenna.png')
def function(image):
    height,width,channels =image.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh + 0.5)
            y=int(j/sw + 0.5)
            emptyImage[i,j]=image[x,y]
    return emptyImage

img=function(image)
cv2.imshow("nearest interp",img)
cv2.imshow("image",image)
cv2.waitKey(0)