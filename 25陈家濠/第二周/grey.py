from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img=cv2.imread("lenna.png")
h,w=img.shape[:2]
img_gray=np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
cv2.imshow("image show gray",img_gray)
cv2.waitKey()

# 二值化

a,b=img_gray.shape
img_two=np.zeros([a,b],img_gray.dtype)
for i in range(a):
    for j in range(b):
        x=img_gray[i,j]
        if x>128:
            img_two[i,j]=255
        else:
            img_two[i,j]=0
cv2.imshow("image show gray",img_two)
cv2.waitKey()
