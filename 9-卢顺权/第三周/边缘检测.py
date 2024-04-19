import cv2
import numpy as np
import  matplotlib.pyplot as plt
img = cv2.imread("LYF.jpeg", 0)
x=cv2.Sobel(img,cv2.CV_16S,1,0) #1代表x方向
y=cv2.Sobel(img,cv2.CV_16S,0,1)
abs_x=cv2.convertScaleAbs(x)
abs_y=cv2.convertScaleAbs(y)
res = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

# cv2.imshow('',abs_x)
# cv2.imshow('',abs_y)
# cv2.imshow('',res)
# cv2.waitKey()
plt.subplot(2,2,1)
plt.imshow(abs_x,cmap='gray')
plt.subplot(2,2,2)
plt.imshow(abs_y,cmap='gray')
plt.subplot(2,2,3)
plt.imshow(res,cmap='gray')
plt.show()