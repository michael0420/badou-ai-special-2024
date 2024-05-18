"""
@author: huangyunxuan

彩色图像的灰度化、二值化
"""
import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

# 灰度化

img = cv2.imread("lenna.png")
w, h, c  = img.shape

print(w,h,c)
sw = 800/w
sh = 800/h
def function(img):
    img_zoom = np.zeros((800,800,c,),np.uint8)s
    for i in range(800):
        for j in range(800):
            x = int(i/sw + 0.5)
            y = int(j/sh + 0.5)
            img_zoom[i,j] = img[x,y]
    return img_zoom


zoom = function(img)
cv2.imshow("img",img)
cv2.imshow("img_zoom",zoom)

cv2.waitKey(0)

