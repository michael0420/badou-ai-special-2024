# 图片最临近插值
import cv2
import numpy as np

def nearest_interp(img):
    height,width,channels = img.shape
    emptyImage = np.zeros((800,800,channels),np.uint8)        # 创建一张尺寸为800*800的空白图片
    sh = 800 / height
    sw = 800 / width

    for i in range(800):
        for j in range(800):
            x = int(i/sh + 0.5)                       # int(),转为整型，使用向下取整。
            y = int(j/sw + 0.5)
            emptyImage[i,j] = img[x,y]                # 遍历空白图片的坐标，把img图片x,y坐标像素值赋到空白图片
    return emptyImage

img = cv2.imread('lenna.png')
emptyImage = nearest_interp(img)
print(emptyImage)
print(emptyImage.shape)

cv2.imshow('nearest interp',emptyImage)
cv2.imshow('image',img)
cv2.waitKey(0)








