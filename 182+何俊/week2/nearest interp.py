import cv2
import numpy as np
def function(img):
    height,width,channels = img.shape#shape函数，用来获取图片的长宽和通道数
    emptyImage = np.zeros((800,800,channels),np.uint8)
    sh = 800/height
    sw = 800/width
    for i in range(800):
        for j in range(800): 
            x = int(i/sh + 0.5)#转为整型，向下取整，+0.5相当于四舍五入
            y = int(j/sw + 0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage

img = cv2.imread(r"E:\AI\CV\second week\work\lenna.png")#读取图像
zoom = function(img)
cv2.imshow("nearest interp",zoom)
cv2.waitKey(0)
