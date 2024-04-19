import cv2
import numpy as  np

img=cv2.imread('F:/PNG/lenna.png')  #读入图片
heigt,width=img.shape[:2]   #获取图片的高和宽
img_gray=np.zeros([heigt,width],img.dtype)   #创建一张和当前大小一样的新图片
for i in  range(heigt):
    for j in range(width):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11 +m[1]*0.59 +m[2]*0.3)   #浮点算法，BGR转换为灰度值
cv2.imshow('Original image',img)
cv2.imshow('Grayscale image',img_gray)
cv2.waitKey(0)
