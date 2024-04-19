import cv2
import numpy as np
img = cv2.imread("lenna.png")
height,width,channels = img.shape
nearest_img = np.zeros((800,800,channels), img.dtype)
#计算扩大倍数
nh = 800/height
nw = 800/width
for i in range(800):
    for j in range(800):
        x=int(i/nh + 0.5)
        y=int(j/nw + 0.5)
        nearest_img[i,j] =img[x,y]

cv2.imshow("nearest_img",nearest_img)
cv2.imshow("image", img)
cv2.waitKey()