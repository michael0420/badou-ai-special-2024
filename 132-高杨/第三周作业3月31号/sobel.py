import cv2
import numpy as np

img = cv2.imread('lenna.png',1)


# sobel 函数求导后会有负值，还有大于255的值，原图是 unit8
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)

# 使用convertScaleAbs（）函数激将其转化为原来的unint8格式
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)


#由于Sobel算子两个方向计算的，最后需要cv2.addWeighted 来分配x，y的权重
target = cv2.addWeighted(absX,0.8,absY,0.2,0)

cv2.imshow('x',absX)
cv2.imshow('y',absY)

cv2.imshow('addweight',absY)
cv2.waitKey(0)