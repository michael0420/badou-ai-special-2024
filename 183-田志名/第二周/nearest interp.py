import numpy as np
import cv2
import matplotlib.pyplot as plt
img=cv2.imread("lenna.png")
h,w,c=img.shape
bigimage=np.zeros([800,800,c],np.uint8)
for i in range(800):
    for j in range(800):
        x=int(i/(800/h)+0.5)
        y=int(j/(800/w)+0.5)
        bigimage[i,j]=img[x,y]
cv2.imshow("bigimage",bigimage)
cv2.imshow("preimage",img)
cv2.waitKey(0)    #waitkey控制着imshow的持续时间，当imshow之后不跟waitkey时，相当于没有给imshow提供时间展示图像，所以只有一个空窗口一闪而过

