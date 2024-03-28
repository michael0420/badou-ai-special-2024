import numpy as np
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

#最邻近插值法 放大
def function(img,height,width):
    h,w,c=img.shape
    emptyImage=np.zeros((height,width,c),np.uint8)
    sh=height/h
    sw=width/w
    for i in range(height):
        for j in range(width):
            x = int(i/sh+0.5)
            y = int(j/sh+0.5)
            emptyImage[i,j] = img[x,y]
    return emptyImage



img = cv2.imread("lenna.png")
#放大
zoom = function(img,800,800)
cv2.imshow("new Image",zoom)

cv2.waitKey(0)