import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.color import rgb2gray
from PIL import Image


img=cv2.imread('D:/lenna.png')  #读入图片
cv2.imshow('Original image',img) #显示原图
heigt,width=img.shape[:2]   #获取图片的高和宽
img_gray=np.zeros([heigt,width],img.dtype)   #创建一张和当前大小一样的新图片
for i in  range(heigt):
    for j in range(width):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11 +m[1]*0.59 +m[2]*0.3)   #浮点算法，BGR转换为灰度值，灰度化
print(img_gray)
cv2.imshow('Grayscale image',img_gray) #显示灰度化后的图片
for x in  range(heigt):
    for y in  range(width):
         n=int(img_gray[x,y])
         if n>150:  #二值化
            img_gray[x,y]=255
         else:
            img_gray[x,y]=0
print(img_gray)
cv2.imshow('Binary image',img_gray)  #显示二值化后的图片
cv2.waitKey(0)

