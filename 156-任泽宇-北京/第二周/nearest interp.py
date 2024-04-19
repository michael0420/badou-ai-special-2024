
import cv2
import numpy as np

img=cv2.imread("lenna.png")

height,width,channels =img.shape
dh = 600
dw = 600
emptyImage=np.zeros((dh,dw,channels),np.uint8)
sh=dh/height
sw=dw/width
for i in range(dh):
    for j in range(dw):
        x=int(i/sh + 0.5)
        y=int(j/sw + 0.5)
        emptyImage[i,j]=img[x,y]

cv2.imshow("image",img)
cv2.imshow("nearest interp",emptyImage)
cv2.waitKey(0)


